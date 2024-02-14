from typing import List, Optional

import albumentations as A
import click
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
import pytorch_toolbelt.losses
import torchmetrics
import lion_pytorch
import pathlib

import trafos
from data import Channel as Ch
from data import KelpTiledDataset, get_train_val_test_masks


GLOBAL_SEED = 42  # DO NOT CHANGE

VALID_CHANNELS = [
    Ch.SWIR,
    Ch.NIR,
    Ch.R,
    Ch.G,
    Ch.B,
    Ch.NDWI_1,
    Ch.NDWI_2,
    Ch.NDVI,
    Ch.GNDVI,
    Ch.NDTI,
    Ch.EVI,
    Ch.CARI,
]


class ConvBlock(nn.Module):
    """Two convolution layers with batch normalization and ReLU activation.

    - Increase the number of channels from `in_c` to `out_c`.
    - Decrease resolution by a factor of 4.
    """
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
    """Apply convolution to input for next layer and return maxpol of it for skip connection"""
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


class UNet(nn.Module):
    def __init__(self, n_ch):
        super().__init__()

        # Encoder
        self.e1 = EncoderBlock(n_ch, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        # Bottleneck
        self.b = ConvBlock(512, 1024)

        # Decoder
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        # Classifier
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()  # kelp cover is a binary mask

    def forward(self, inputs):
        # Encoder
        s1, p = self.e1(inputs)
        s2, p = self.e2(p)
        s3, p = self.e3(p)
        s4, p = self.e4(p)

        # Bottleneck
        b = self.b(p)

        # Decoder
        d = self.d1(b, s4)
        d = self.d2(d, s3)
        d = self.d3(d, s2)
        d = self.d4(d, s1)

        # Classifier
        outputs = self.outputs(d)
        outputs = outputs.squeeze(1)  # remove channel dimension since we only have one channel
        return torch.sigmoid(outputs)


class LitUNet(L.LightningModule):
    def __init__(self, n_ch: int):
        super().__init__()
        self.save_hyperparameters()

        self.model = UNet(n_ch=n_ch)
        self.crit = pytorch_toolbelt.losses.DiceLoss(mode="binary", from_logits=False)
        self.dice = torchmetrics.Dice()

    def _shared_step(self, batch, batch_idx, prefix: str):
        x, y = batch
        y_hat = self.model(x)
        loss = self.crit(y_hat, y)
        self.log(f"{prefix}_loss", loss, sync_dist=True)
        self.log(f"{prefix}_dice", self.dice(y_hat, y.int()), sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = lion_pytorch.Lion(self.parameters(), lr=1e-4, weight_decay=1e-1)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.95)
        return [optimizer], [lr_scheduler]


def get_dataset(use_channels: Optional[List[int]], random_seed: int):
    def drop_channels(img, mask):
        """Only keep specified channels. Expecting channel as last dimension."""
        img = img[:, :, use_channels]
        return img, mask

    def apply_train_trafos(ds: KelpTiledDataset) -> None:
        aug_pipeline = A.Compose(
            [
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ]
        )

        def apply_aug(img, mask):
            res = aug_pipeline(image=img, mask=mask)
            return res["image"], res["mask"]

        # Drop channels if requested
        if use_channels is not None:
            ds.add_transform(drop_channels)

        ds.add_transform(trafos.xr_to_np)
        ds.add_transform(apply_aug)  # Random augmentation only during training!
        ds.add_transform(trafos.channel_first)
        ds.add_transform(trafos.to_tensor)

    def apply_infer_trafos(ds: KelpTiledDataset) -> None:
        if use_channels is not None:
            ds.add_transform(drop_channels)

        ds.add_transform(trafos.xr_to_np)
        ds.add_transform(trafos.channel_first)
        ds.add_transform(trafos.to_tensor)

    ds_kwargs = {
        "img_nc_path": "data_ncf/train_imgs_fe.nc",
        "mask_nc_path": "data_ncf/train_masks.ncf",
        "n_rand_tiles": 25,
        "tile_size": 64,
        "random_seed": random_seed,
    }
    ds = KelpTiledDataset(**ds_kwargs)

    # Split data into train/val/test
    mask_train, mask_val, mask_test = get_train_val_test_masks(len(ds.imgs), random_seed=random_seed)

    # Load dataset without outlier filter
    ds_train = KelpTiledDataset(**ds_kwargs, sample_mask=mask_train)
    apply_train_trafos(ds_train)

    ds_val = KelpTiledDataset(**ds_kwargs, sample_mask=mask_val)
    ds_test = KelpTiledDataset(**ds_kwargs, sample_mask=mask_test)
    apply_infer_trafos(ds_val)
    apply_infer_trafos(ds_test)

    return ds_train, ds_val, ds_test


def get_loaders(use_channels: Optional[List[int]], random_seed: int, **loader_kwargs):
    ds_train, ds_val, ds_test = get_dataset(use_channels=use_channels, random_seed=random_seed)

    ds_train.load()
    ds_val.load()
    ds_test.load()

    # Shuffle data for training
    train_loader = torch.utils.data.DataLoader(ds_train, shuffle=True, **loader_kwargs)

    # Normal sampling for val and test
    val_loader = torch.utils.data.DataLoader(ds_val, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(ds_test, **loader_kwargs)

    return train_loader, val_loader, test_loader


def train(*, n_ch: Optional[int],  i_member: int, i_device: int, ens_root: str):
    # Make sure ens_root exists and is empty
    ens_root = pathlib.Path(ens_root)
    ens_root.mkdir(exist_ok=True)

    # Init rng with global seed to get rng for this member
    rng = np.random.default_rng(GLOBAL_SEED)
    random_seed = None
    for _ in range(i_member + 1):
        random_seed = rng.integers(0, 2**32 - 1)

    # Now create new rng for this member
    rng = np.random.default_rng(random_seed)
    random_seed = rng.integers(0, 2**32 - 1)

    # Select channel subset
    if n_ch is None:
        # Default
        use_channels = [0, 1, 2, 6, 8, 9, 10, 11]
        n_ch = len(use_channels)
    else:
        # Random choice
        use_channels = rng.choice(VALID_CHANNELS, size=n_ch, replace=False)
    use_channels = np.array(use_channels)

    print(f"Member {i_member} uses channels: {use_channels}.")

    train_loader, val_loader, test_loader = get_loaders(
        use_channels=use_channels, num_workers=0, batch_size=1024, random_seed=random_seed, pin_memory=True
    )

    # Save best models
    ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_dice",
        mode="max",
        dirpath=ens_root,
        filename=f"seg_{i_member}_" + "-".join(use_channels.astype(str)) + "_{epoch:02d}_{val_dice:.2f}",
    )

    # Train
    model = LitUNet(n_ch=n_ch)
    trainer = L.Trainer(
        devices=[i_device],
        max_epochs=15,
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
    trainer = L.Trainer(devices=[i_device], logger=None)
    trainer.test(model, dataloaders=test_loader)


@click.command()
@click.option("--ens_root", type=str, default="ens_seg/dev")
@click.argument("i_member", type=int)
@click.argument("i_device", type=int)
def train_cli(ens_root: str, i_member: int, i_device: int):
    train(n_ch=3, i_member=i_member, i_device=i_device, ens_root=ens_root)


if __name__ == "__main__":
    train_cli()
