from typing import List, Optional

import albumentations as A
import click
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lion_pytorch import Lion
import pytorch_toolbelt.losses
import pathlib
import torchmetrics

import trafos
import shared
from data import Channel as Ch
from data import KelpTiledDataset, get_train_val_test_masks


class BinaryClfCNN(nn.Module):
    def __init__(self, n_ch: int, fc_size: int, p_dropout: float):
        res_in = 64
        ch1, ch2 = 64, 128

        super(BinaryClfCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=n_ch, out_channels=ch1, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=p_dropout),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=ch1, out_channels=ch2, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=p_dropout),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(ch2 * (res_in // 4) * (res_in // 4), fc_size),
            nn.Dropout(p=p_dropout),
            nn.ReLU(),
            nn.Linear(fc_size, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        # x = x.squeeze(1)
        return x


class LitBinaryClf(L.LightningModule):
    def __init__(self, n_ch: int, fc_size=128, p_dropout=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.model = BinaryClfCNN(n_ch=n_ch, fc_size=fc_size, p_dropout=p_dropout)
        self.crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10))  # ~13.3% of samples have kelp
        # self.crit = pytorch_toolbelt.losses.DiceLoss(mode="binary", from_logits=True)
        self.dice = torchmetrics.Dice()

    def _shared_eval_step(self, batch, batch_idx, prefix):
        x, y = batch
        y = y.unsqueeze(1)

        y_hat = self.model(x)
        loss = self.crit(y_hat, y.float())
        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_dice", self.dice(torch.sigmoid(y_hat), y))

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        return Lion(self.parameters(), lr=1e-4)


def apply_train_trafos(ds: KelpTiledDataset, mode: str) -> None:
    aug_pipeline = A.Compose(
        [
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ]
    )

    def apply_aug(img, mask):
        if mode == "binary":
            res = aug_pipeline(image=img)
            return res["image"], mask
        else:
            res = aug_pipeline(image=img, mask=mask)
            return res["image"], res["mask"]

    if mode == "binary":
        ds.add_transform(trafos.to_binary_kelp)

    ds.add_transform(trafos.xr_to_np)
    ds.add_transform(apply_aug)  # Random augmentation only during training!
    ds.add_transform(trafos.channel_first)
    ds.add_transform(trafos.to_tensor)


def apply_infer_trafos(ds: KelpTiledDataset, mode: str) -> None:
    if mode == "binary":
        ds.add_transform(trafos.to_binary_kelp)

    ds.add_transform(trafos.xr_to_np)
    ds.add_transform(trafos.channel_first)
    ds.add_transform(trafos.to_tensor)


def get_dataset(use_channels: Optional[List[int]], split_seed: int, tile_seed: int, mode: str):
    """Get dataset with channel subset and split into train/val/test.
    Two seeds are used in an ensemble case:
    - `split_seed` should be the same for all members to get reproducible (independent) train/val/test splits.
    - `tile_seed` should be different for each member to get different random tiles for each member.
    """
    ds_kwargs = {
        "img_nc_path": "data_ncf/train_imgs_fe.nc",
        "mask_nc_path": "data_ncf/train_masks.ncf",
        "n_rand_tiles": 25,
        "tile_size": 64,
        "random_seed": tile_seed,
        "use_channels": use_channels,
    }
    ds = KelpTiledDataset(**ds_kwargs)

    # Split data into train/val/test
    mask_train, mask_val, mask_test = get_train_val_test_masks(len(ds.imgs), random_seed=split_seed)

    # Load dataset without outlier filter
    ds_train = KelpTiledDataset(**ds_kwargs, sample_mask=mask_train)
    apply_train_trafos(ds_train, mode=mode)

    ds_val = KelpTiledDataset(**ds_kwargs, sample_mask=mask_val)
    ds_test = KelpTiledDataset(**ds_kwargs, sample_mask=mask_test)
    apply_infer_trafos(ds_val, mode=mode)
    apply_infer_trafos(ds_test, mode=mode)

    return ds_train, ds_val, ds_test


def get_loaders(use_channels: Optional[List[int]], split_seed: int, tile_seed: int, mode: str, **loader_kwargs):
    ds_train, ds_val, ds_test = get_dataset(use_channels=use_channels, split_seed=split_seed, tile_seed=tile_seed, mode=mode)

    # Load data to RAM for fast training
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
    ens_root.mkdir(exist_ok=True, parents=True)

    # Get random seed for this member
    random_seed = shared.get_local_seed(i_member)

    # Select channel subset
    use_channels, n_ch = shared.get_channel_subset(n_ch, random_seed)
    print(f"Member {i_member} uses channels: {use_channels}.")

    train_loader, val_loader, test_loader = get_loaders(
        use_channels=use_channels, split_seed=shared.GLOBAL_SEED, tile_seed=random_seed, mode="binary",
        num_workers=0, batch_size=1024, pin_memory=True
    )

    # Save best models
    ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_dice",
        mode="max",
        dirpath=ens_root,
        filename=f"clf_{i_member}_" + "-".join(use_channels.astype(str)) + "_{epoch:02d}_{val_dice:.2f}",
    )

    # Train
    model = LitBinaryClf(n_ch=n_ch)
    trainer = L.Trainer(
        devices=[i_device],
        max_epochs=30,
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
@click.option("--ens_root", type=str, default="ens_clf/dev")
@click.argument("i_member", type=int)
@click.argument("i_device", type=int)
def train_cli(ens_root: str, i_member: int, i_device: int):
    train(n_ch=3, i_member=i_member, i_device=i_device, ens_root=ens_root)


if __name__ == "__main__":
    train(n_ch=3, i_member=0, i_device=0, ens_root="ens_clf/dev")
    # train_cli([0, 0])
