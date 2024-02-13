from typing import List, Optional

import albumentations as A
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lion_pytorch import Lion
from torchmetrics import Accuracy

import trafos
from data import KelpTiledDataset, get_train_val_test_masks
from data import Channel as Ch


class BinaryClfCNN(nn.Module):
    def __init__(self, n_ch: int, fc_size: int, p_dropout: float):
        super(BinaryClfCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=n_ch, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=p_dropout),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=p_dropout),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 16 * 16, fc_size), nn.Dropout(p=p_dropout), nn.ReLU(), nn.Linear(fc_size, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        x = x.squeeze(1)
        return x


class LitBinaryClf(L.LightningModule):
    def __init__(self, n_ch: int, fc_size=128, p_dropout=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.model = BinaryClfCNN(n_ch=n_ch, fc_size=fc_size, p_dropout=p_dropout)
        self.accuracy = Accuracy(task="binary", threshold=0.5)

    def _shared_eval_step(self, batch, batch_idx, prefix):
        x, y = batch
        y_hat = self.model(x)

        loss = nn.BCEWithLogitsLoss()(y_hat, y.type_as(y_hat))
        acc = self.accuracy(torch.round(y_hat), y)
        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_acc", acc, prog_bar=True)
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
        ds.add_transform(trafos.to_binary_kelp)
        ds.add_transform(trafos.to_tensor)

    def apply_infer_trafos(ds: KelpTiledDataset) -> None:
        if use_channels is not None:
            ds.add_transform(drop_channels)

        ds.add_transform(trafos.xr_to_np)
        ds.add_transform(trafos.channel_first)
        ds.add_transform(trafos.to_binary_kelp)
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

    # Shuffle data for training
    train_loader = torch.utils.data.DataLoader(ds_train, shuffle=True, **loader_kwargs)

    # Normal sampling for val and test
    val_loader = torch.utils.data.DataLoader(ds_val, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(ds_test, **loader_kwargs)

    return train_loader, val_loader, test_loader


def train(i_member: int, i_device: int, n_ensemble: int = 25):
    GLOBAL_SEED = 42  # DO NOT CHANGE

    N_USE_CHANNELS = 3
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

    # Init rng with global seed to get rng for this member
    rng = np.random.default_rng(GLOBAL_SEED)
    random_seed = rng.integers(0, 2**32 - 1, size=n_ensemble)[i_member]

    # Now create new rng for this member
    rng = np.random.default_rng(random_seed)
    random_seed = rng.integers(0, 2**32 - 1)

    # Select channel subset
    use_channels = rng.choice(VALID_CHANNELS, size=N_USE_CHANNELS, replace=False)
    print(f"Member {i_member} uses channels: {use_channels}.")

    train_loader, val_loader, test_loader = get_loaders(
        use_channels=use_channels, num_workers=32, batch_size=128, random_seed=random_seed, prefetch_factor=10
    )

    # Train
    model = LitBinaryClf(n_ch=N_USE_CHANNELS)
    trainer = L.Trainer(
        devices=[i_device],
        max_epochs=1,
        log_every_n_steps=10,
        callbacks=[
            # EarlyStopping(monitor="val_dice", patience=3, mode="max"),
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


if __name__ == "__main__":
    train(1, 0)
