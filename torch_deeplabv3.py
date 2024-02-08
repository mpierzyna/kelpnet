import albumentations as A
import lightning as L
import torch
from torch import nn
import torch.utils.data
import rasterio
from torchvision.models.segmentation import deeplabv3_resnet101
import pytorch_toolbelt.losses
import lion_pytorch
import torchmetrics
import warnings
import pandas as pd
import logging
import numpy as np

from data import KelpDataset, split_train_test_val2
import trafos


torch.set_float32_matmul_precision("high")
# Reading dataset raises warnings, which are anoying
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Setup logging
logger = logging.getLogger("kelp")
logging.basicConfig(level=logging.INFO)
logging.getLogger("rasterio").setLevel(logging.ERROR)


class DeepLabV3(nn.Module):
    def __init__(self, n_ch: int):
        super().__init__()

        # Load a preconfigured DeepLabV3 with one output class
        self.model = deeplabv3_resnet101(num_classes=1)

        # Modify the first convolution layer to accept n_ch channels
        self.model.backbone.conv1 = nn.Conv2d(n_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        # Forward pass through the modified DeepLabV3 model
        output = self.model(x)["out"]

        # remove channel dimension since we only have one channel
        output = output.squeeze(1)

        # Apply sigmoid activation for binary classification
        return torch.sigmoid(output)


class LitDeepLabV3(L.LightningModule):
    def __init__(self, n_ch: int, ens_prediction: bool):
        super().__init__()
        self.model = DeepLabV3(n_ch=n_ch)
        self.crit = pytorch_toolbelt.losses.DiceLoss(mode="binary", from_logits=False)

        # Additional metrics
        self.dice = torchmetrics.Dice()

        # Ensemble prediction flag
        self.ens_prediction = ens_prediction

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x_valid = torch.where(x[:, 0, :, :] < 0, 0, 1)  # mask for NaN values, to zero them for loss computation

        y_hat = self.model(x)
        loss = self.crit(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_dice", self.dice(y_hat, y.int()))

        return loss

    def _ens_predict(self, x):
        """Make ensemble prediction by rotating BATCH x 90 degrees and averaging the predictions"""
        y_hat_ens = []
        for k in [0, 1, 2, 3]:
            # Make prediction on rotated x and rotate back (First x not yet rotated)
            y_hat = self.model(x)
            y_hat = torch.rot90(y_hat, k=-k, dims=[1, 2])
            y_hat_ens.append(y_hat)

            # Rotate 90 degrees for next iteration. Happens everytime (k=1) -> adds up
            x = torch.rot90(x, k=1, dims=[2, 3])

        # Stack and return statistics
        y_hat_ens = torch.stack(y_hat_ens)
        return y_hat_ens.mean(dim=0), y_hat_ens.std(dim=0)

    def _shared_eval_step(self, batch, batch_idx, prefix: str):
        x, y = batch

        if self.ens_prediction:
            y_hat, y_hat_std = self._ens_predict(x)
            
            # Compute and log std on non-zero values only
            y_hat_std = y_hat_std.flatten()
            y_hat_std = y_hat_std[y_hat_std > 0]
            y_hat_nz_std = torch.std(y_hat_std)
            self.log(f"{prefix}_nz_std", y_hat_nz_std)
        else:
            y_hat = self.model(x)

        loss = self.crit(y_hat, y)
        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_dice", self.dice(y_hat, y.int()))

        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, prefix="test")

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        if self.ens_prediction:
            y_hat, y_hat_std = self._ens_predict(x)
        else:
            y_hat = self.model(x)
        y_hat = y_hat > .5
        return y_hat

    def configure_optimizers(self):
        opt = lion_pytorch.Lion(self.parameters(), lr=1e-4)
        return opt


def apply_train_trafos(ds: KelpDataset) -> None:
    aug_pipeline = A.Compose(
        [
            A.HorizontalFlip(),
            # A.VerticalFlip(),
            # A.ChannelDropout(p=.1, fill_value=0),
        ]
    )

    def apply_aug(img, mask):
        res = aug_pipeline(image=img, mask=mask)
        return res["image"], res["mask"]

    ds.add_transform(trafos.add_rs_indices)
    ds.add_transform(apply_aug)  # Random augmentation only during training!
    ds.add_transform(trafos.channel_first)
    ds.add_transform(trafos.to_tensor)


def apply_infer_trafos(ds: KelpDataset) -> None:
    ds.add_transform(trafos.add_rs_indices)
    ds.add_transform(trafos.channel_first)
    ds.add_transform(trafos.to_tensor)


def get_dataset():
    # Init dataset but don't apply trafos
    ds = KelpDataset(img_dir="data_inpainted/train_satellite/", mask_dir="data/train_kelp/")

    # Split into sub datasets and apply trafos
    ds_train, ds_val, ds_test = split_train_test_val2(ds)
    apply_train_trafos(ds_train)

    apply_infer_trafos(ds_val)
    apply_infer_trafos(ds_test)

    return ds_train, ds_val, ds_test


def get_loaders(kf_weighing: bool, **loader_kwargs):
    ds_train, ds_val, ds_test = get_dataset()

    if kf_weighing:
        # Use kelp fraction as sampling weights
        df_quality = pd.read_csv("quality.csv", index_col=0)
        kf = (df_quality["kelp_fraction"] + 0.05) / 0.2
        kf[kf > 1] = 0

        # Train loader with weighted random sampling
        w = kf[ds_train.dir_mask].values
        train_sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        train_loader = torch.utils.data.DataLoader(ds_train, sampler=train_sampler, **loader_kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(ds_train, shuffle=True, **loader_kwargs)

    # Normal sampling for val and test
    val_loader = torch.utils.data.DataLoader(ds_val, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(ds_test, **loader_kwargs)

    return train_loader, val_loader, test_loader


def test_loaders():
    """Iterate through loaders to check if they work"""
    train_loader, val_loader, test_loader = get_loaders(num_workers=1, batch_size=32, kf_weighing=False)

    # Test loaders
    print("Testing loaders")
    for loader in [train_loader, val_loader, test_loader]:
        for x, y in loader:
            print(x.shape, y.shape)

    print("Done. Exiting")

    import sys

    sys.exit(0)


if __name__ == "__main__":
    # test_loaders()

    train_loader, val_loader, test_loader = get_loaders(num_workers=8, batch_size=32, kf_weighing=False)

    # Train
    model = LitDeepLabV3(n_ch=11)
    trainer = L.Trainer(
        devices=1,
        max_epochs=50,
        log_every_n_steps=10,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # New trainer on just one device
    trainer.test(ckpt_path="best", dataloaders=test_loader)
