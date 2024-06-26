import logging
import pathlib
import warnings
from typing import List, Optional

import albumentations as A
import click
import lightning as L
import lion_pytorch
import numpy as np
import pandas as pd
import pytorch_toolbelt.losses
import rasterio
import torch
import torch.nn.functional as F
import torch.utils.data
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from torch import nn
from torchvision.models.segmentation import (deeplabv3_resnet50,
                                             deeplabv3_resnet101)
from torchvision.models.segmentation import deeplabv3 as tv_dlv3
from torchvision.models import resnet18, resnet34
from torchvision.models._utils import IntermediateLayerGetter

import shared
import trafos
from data import KelpNCDataset, get_train_val_test_masks

torch.set_float32_matmul_precision("high")
# Reading dataset raises warnings, which are anoying
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Setup logging
logger = logging.getLogger("kelp")
logging.basicConfig(level=logging.INFO)
logging.getLogger("rasterio").setLevel(logging.ERROR)


class DeepLabV3(nn.Module):
    def __init__(self, n_ch: int, upsample_size: int):
        super().__init__()

        self.n_ch = n_ch
        self.upsample_size = upsample_size

        """
        # Use ResNet backbone
        backbone = resnet34()
        if n_ch != 3:
            # If channels != 3, modified first layer to accept n_ch channels 
            backbone.conv1 = nn.Conv2d(n_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "out"})
        self.backbone = backbone

        # Set Deeplab head
        self.classifier = tv_dlv3.DeepLabHead(512, 1)

        # Combine backbone and head
        self.model = tv_dlv3.DeepLabV3(self.backbone, self.classifier)
        """

        # Load a preconfigured DeepLabV3 with one output class
        self.model = deeplabv3_resnet50(num_classes=1)

        # Modify the first convolution layer to accept n_ch channels
        self.model.backbone.conv1 = nn.Conv2d(n_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        # Upscale x before backbone pass
        input_shape = x.shape[-2:]
        x = F.interpolate(x, size=(self.upsample_size, self.upsample_size), mode="bilinear", align_corners=False)

        # Backbone pass
        x = self.model.backbone(x)["out"]

        # Classifier pass (DeepLabV3 head)
        x = self.model.classifier(x)

        # Interpolate to original size
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        # remove channel dimension since we only have one channel
        x = x.squeeze(1)

        # Apply sigmoid activation for binary classification
        return torch.sigmoid(x)


class LitDeepLabV3(L.LightningModule):
    def __init__(self, n_ch: int, upsample_size: int, ens_prediction: bool, lr: float, weight_decay: float, lr_gamma: float):
        super().__init__()
        self.model = DeepLabV3(n_ch=n_ch, upsample_size=upsample_size)
        self.crit = pytorch_toolbelt.losses.DiceLoss(mode="binary", from_logits=False)

        # Additional metrics
        self.dice = torchmetrics.Dice()

        # Store arguments (should also be accessible via self.hparams)
        self.save_hyperparameters()
        self.n_ch = n_ch
        self.upsample_size = upsample_size
        self.ens_prediction = ens_prediction
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x_valid = torch.where(x[:, 0, :, :] < 0, 0, 1)  # mask for NaN values, to zero them for loss computation

        y_hat = self.model(x)
        loss = self.crit(y_hat, y)
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_dice", self.dice(y_hat, y.int()), sync_dist=True)

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
            y_hat_nz_std = torch.mean(y_hat_std)
            self.log(f"{prefix}_nz_std", y_hat_nz_std, sync_dist=True)
        else:
            y_hat = self.model(x)

        loss = self.crit(y_hat, y)
        self.log(f"{prefix}_loss", loss, sync_dist=True)
        self.log(f"{prefix}_dice", self.dice(y_hat, y.int()), sync_dist=True, prog_bar=True)

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
        y_hat = y_hat > 0.5
        return y_hat

    def configure_optimizers(self):
        optimizer = lion_pytorch.Lion(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [lr_scheduler]


def apply_train_trafos(ds: KelpNCDataset) -> None:
    aug_pipeline = A.Compose(
        [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # A.RandomSizedCrop(min_max_height=(200, 350), height=350, width=350, w2h_ratio=1),
            # A.PixelDropout(dropout_prob=0.01, per_channel=True)
        ]
    )

    def apply_aug(img, mask):
        res = aug_pipeline(image=img, mask=mask)
        return res["image"], res["mask"]

    ds.add_transform(trafos.xr_to_np)
    ds.add_transform(apply_aug)  # Random augmentation only during training!
    ds.add_transform(trafos.channel_first)
    ds.add_transform(trafos.to_tensor)


def apply_infer_trafos(ds: KelpNCDataset) -> None:
    ds.add_transform(trafos.xr_to_np)
    ds.add_transform(trafos.channel_first)
    ds.add_transform(trafos.to_tensor)


def get_dataset(use_channels: Optional[List[int]], random_seed: int):
    # Load dataset without outlier filter
    ds_kwargs = {
        "img_nc_path": "data_ncf/train_imgs_fe.nc",
        "mask_nc_path": "data_ncf/train_masks.ncf",
        "use_channels": use_channels,
    }
    ds = KelpNCDataset(**ds_kwargs)
    mask_train, mask_val, mask_test = get_train_val_test_masks(len(ds), random_seed=random_seed)

    """
    all_tiles = ds.tile_ids

    # Perform train/val/test split based on number of valid tiles only
    is_outlier = pd.read_csv("is_outlier.csv")["0"]
    valid_tiles = all_tiles[~is_outlier]
    mask_train, mask_val, mask_test = get_train_val_test_masks(len(valid_tiles), random_seed=random_seed)

    # Loading requires mask of full length of dataset, so we fuse train/val/test masks and outlier masks
    mask_train = np.isin(all_tiles, valid_tiles[mask_train])
    mask_val = np.isin(all_tiles, valid_tiles[mask_val])
    mask_test = np.isin(all_tiles, valid_tiles[mask_test])
    """

    # Now perform loading on split and filtered dataset
    ds_train = KelpNCDataset(**ds_kwargs, sample_mask=mask_train)
    apply_train_trafos(ds_train)

    ds_val = KelpNCDataset(**ds_kwargs, sample_mask=mask_val)
    ds_test = KelpNCDataset(**ds_kwargs, sample_mask=mask_test)
    apply_infer_trafos(ds_val)
    apply_infer_trafos(ds_test)

    return ds_train, ds_val, ds_test


def get_submission_dataset():
    ds = KelpNCDataset(img_nc_path="data_ncf/test_imgs_fe.nc", mask_nc_path=None)
    apply_infer_trafos(ds)
    return ds


def get_loaders(*, use_channels: Optional[List[int]], kf_weighing: bool, random_seed: int, **loader_kwargs):
    ds_train, ds_val, ds_test = get_dataset(use_channels=use_channels, random_seed=random_seed)

    if kf_weighing:
        # Use kelp fraction as sampling weights
        df_quality = pd.read_csv("quality.csv", index_col=0)
        kf = df_quality["kelp_fraction"].clip(None, .1).to_numpy()
        w = 1 + np.power(kf, 1/5)  # Weighting

        # Train loader with weighted random sampling
        w = w[ds_train.sample_mask]
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


def train(*, n_ch: Optional[int], i_member: int, ens_dir: str):
    # Make sure ens_root exists and is empty
    ens_dir = pathlib.Path(ens_dir)
    ens_dir.mkdir(exist_ok=True, parents=True)

    # Get random seed for this member
    random_seed = shared.get_local_seed(i_member)

    # Select channel subset
    use_channels, n_ch = shared.get_channel_subset(n_ch, random_seed)
    print(f"Member {i_member} uses channels: {use_channels}.")

    # Setup loaders
    train_loader, val_loader, test_loader = get_loaders(
        use_channels=use_channels, kf_weighing=False, random_seed=random_seed,
        num_workers=8, batch_size=32,
    )

    # Save best models
    ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_dice",
        mode="max",
        dirpath=ens_dir,
        filename=f"dlv3_{i_member}_" + "-".join(use_channels.astype(str)) + "_{epoch:02d}_{val_dice:.2f}",
    )

    # Train
    model = LitDeepLabV3(n_ch=n_ch, ens_prediction=True, upsample_size=512,
                         lr=5e-4, lr_gamma=0.75, weight_decay=1e-1)
    trainer = L.Trainer(
        devices=[1, 2],
        max_epochs=20,
        log_every_n_steps=10,
        callbacks=[
            ckpt_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # New trainer on just one device
    trainer = L.Trainer(devices=1, logger=False)
    trainer.test(model, dataloaders=test_loader)


@click.command()
@click.option("--ens_dir", type=str, default="ens_dlv3/dev")
@click.argument("i_member", type=int)
def train_cli(ens_dir: str, i_member: int):
    train(n_ch=3, i_member=i_member, ens_dir=ens_dir)


if __name__ == "__main__":
    # test_loaders()
    train_cli()
