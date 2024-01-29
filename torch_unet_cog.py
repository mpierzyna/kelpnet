import enum
import warnings

import lightning as L
import numpy as np
import pandas as pd
import rasterio
import torch
from lion_pytorch import Lion
from torch import nn
import torchmetrics

import trafos
from data import MultiTaskKelpDataset, split_train_test_val

torch.set_float32_matmul_precision("high")
# Reading dataset raises warnings, which are anoying
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class Task(enum.IntEnum):
    SEGMENTATION = 1
    REGRESSION = 2


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class MultiTaskUNet(nn.Module):
    def __init__(self, n_ch, n_regr_out):
        super().__init__()

        # Encoder (halfs image dimensions, doubles channels)
        self.e1 = EncoderBlock(n_ch, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        # Bottleneck (only doubles channels)
        self.b = ConvBlock(512, 1024)

        # Task 1: Segmentation Decoder
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        # Task 1: Segmentation Classifier
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()  # kelp cover is a binary mask

        # Task 2: Regression
        self.regr = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_regr_out),
        )

    def forward(self, inputs, task: Task):
        # Encoder (shared)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # Bottleneck (shared)
        b = self.b(p4)

        if task == Task.SEGMENTATION:
            # Decoder
            d1 = self.d1(b, s4)
            d2 = self.d2(d1, s3)
            d3 = self.d3(d2, s2)
            d4 = self.d4(d3, s1)

            # Classifier
            # No sigmoid here, because we use BCEWithLogitsLoss
            outputs = self.outputs(d4)
            outputs = outputs.squeeze(1)  # remove channel dimension since we only have one channel
            return outputs
        elif task == Task.REGRESSION:
            # Regression
            return self.regr(b)
        else:
            raise ValueError(f"Unknown task {task}")


class LitMTUNet(L.LightningModule):
    def __init__(self, n_ch, n_regr_out):
        super().__init__()
        self.model = MultiTaskUNet(n_ch=n_ch, n_regr_out=n_regr_out)
        self.criterion_segm = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([40]))  # Kelp is rare, so we weight it higher.
        self.criterion_regr = nn.MSELoss()
        self.dice =  torchmetrics.Dice()

    def _shared_step(self, batch, prefix: str):
        x, (y_seg, y_rgr) = batch
        x_valid = torch.where(x[:, 0, :, :] < 0, 0, 1)  # mask for NaN values, to zero them for loss computation

        # Segmentation
        y_hat_seg = self.model(x, task=Task.SEGMENTATION)
        if prefix == "train":
            loss_seg = self.criterion_segm(y_hat_seg * x_valid, y_seg * x_valid)
        else:
            loss_seg = self.criterion_segm(y_hat_seg, y_seg)
        self.log(f"{prefix}_loss_seg", loss_seg)
        self.log(f"{prefix}_dice_seg", self.dice(self.model.sigmoid(y_hat_seg), y_seg.int()))

        # Regression
        y_hat_rgr = self.model(x, task=Task.REGRESSION)
        loss_rgr = self.criterion_regr(y_hat_rgr, y_rgr)
        self.log(f"{prefix}_loss_regr", loss_rgr)

        # Combine losses
        w_seg = 0.5
        w_rgr = 1 - w_seg
        loss = w_seg * loss_seg + w_rgr * loss_rgr
        self.log(f"{prefix}_loss_joint", loss)
        return loss

    def training_step(self, batch, batch_idx):
        """For training, we alternate between tasks per batch"""
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat_seg = self.model(x, task=Task.SEGMENTATION)
        y_hat_seg = torch.sigmoid(y_hat_seg)  # Now apply sigmoid because inference

        y_hat_rgr = self.model(x, task=Task.REGRESSION)

        return y_hat_seg, y_hat_rgr

    def configure_optimizers(self):
        optimizer = Lion(self.parameters(), lr=.5e-3, weight_decay=1e-2)
        return optimizer

    @classmethod
    def _apply_img_trafos(cls, ds: MultiTaskKelpDataset) -> None:
        """Apply trafos to images and masks"""
        ds.add_transform(trafos.add_rs_indices)
        ds.add_transform(trafos.downsample)
        ds.add_transform(trafos.channel_first)
        ds.add_transform(trafos.to_tensor)
        ds.add_transform(lambda X, y: (X, y.float()))  # y is float for BCELoss

    @classmethod
    def _apply_cog_trafos(cls, ds: MultiTaskKelpDataset) -> None:
        """Apply COG trafos"""

        def random_cog(cog):
            if np.isnan(cog[0]):
                # Randomly set x, y, and inertia if no Kelp is present
                cog[[0, 1, 2]] = np.random.uniform(0, 1, size=3)
                cog[2] = cog[2] * 1.4  # scale because inertia is scaled with log10 median
                return cog
            else:
                return cog

        ds.add_regr_transform(random_cog)
        ds.add_regr_transform(lambda y_cog: torch.tensor(y_cog[[0, 1]], dtype=torch.float32))  # only x,y for now

    @classmethod
    def apply_train_trafos(cls, ds: MultiTaskKelpDataset) -> None:
        """Add transformations that are to training dataset"""
        cls._apply_img_trafos(ds)
        cls._apply_cog_trafos(ds)

    @classmethod
    def apply_val_trafos(cls, ds: MultiTaskKelpDataset) -> None:
        """Add transformations that are to validation dataset (no random transformations!)"""
        cls.apply_train_trafos(ds)

    @classmethod
    def apply_test_trafos(cls, ds: MultiTaskKelpDataset) -> None:
        """Add transformations that are to test/prediction dataset (no random transformations!)"""
        cls.apply_train_trafos(ds)


def get_dataset():
    # Load data and add trafos
    quality_df = pd.read_csv("quality.csv")
    ds = MultiTaskKelpDataset(img_dir="data/train_satellite/", mask_dir="data/train_kelp/", cog_path="data/kelp_cog.csv")
    # ds = MultiTaskKelpDataset(img_dir="data/train_satellite/", mask_dir="data/train_kelp/",
    #                           dir_mask=quality_df["nan_fraction"] != 0, cog_path="data/kelp_cog.csv")
    LitMTUNet.apply_train_trafos(ds)

    # Split data
    ds_train, ds_val, ds_test = split_train_test_val(ds)
    return ds_train, ds_val, ds_test


if __name__ == "__main__":
    BATCH_SIZE = 32
    ds_train, ds_val, ds_test = get_dataset()
    train_loader = torch.utils.data.DataLoader(ds_train, num_workers=8, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(ds_val, num_workers=8, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(ds_test, num_workers=8, batch_size=BATCH_SIZE)

    # Train
    model = LitMTUNet(n_ch=11, n_regr_out=2)
    trainer = L.Trainer(
        devices=1,
        # limit_train_batches=256,  # number of total batches into which dataset is split
        max_epochs=15,
        # callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
        log_every_n_steps=10,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # New trainer on just one device
    trainer.test(ckpt_path="best", dataloaders=test_loader)
