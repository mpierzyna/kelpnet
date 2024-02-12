import albumentations as A
import lightning as L
import torch
import torch.nn as nn
from lion_pytorch import Lion
from torchmetrics import Accuracy
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor

from data import KelpTiledDataset, get_train_val_test_masks
import trafos


class BinaryClfCNN(nn.Module):
    def __init__(self, n_ch: int, fc_size: int):
        super(BinaryClfCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=n_ch, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 16 * 16, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


class LitBinaryClf(L.LightningModule):
    def __init__(self, n_ch: int, fc_size=128):
        super().__init__()
        self.model = BinaryClfCNN(n_ch=n_ch, fc_size=fc_size)
        self.accuracy = Accuracy(task="binary", threshold=0.5)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = nn.BCEWithLogitsLoss()(y_hat, y.unsqueeze(1).type_as(y_hat))
        acc = self.accuracy(torch.round(y_hat), y.unsqueeze(1))
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Lion(self.parameters(), lr=1e-4)


def drop_channels(img, mask):
    """Drop channels with low importance from img.
    Expects channels in last dimension.
    """
    # _, _, n_ch = img.shape
    # to_drop = [Ch.R, Ch.G, Ch.B, Ch.IS_CLOUD]
    # to_keep = [ch for ch in range(n_ch) if ch not in to_drop]
    to_keep = [0, 1, 2, 6, 8, 9, 10, 11]
    img = img[:, :, to_keep]
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

    ds.add_transform(drop_channels)
    ds.add_transform(trafos.xr_to_np)
    ds.add_transform(apply_aug)  # Random augmentation only during training!
    ds.add_transform(trafos.channel_first)
    ds.add_transform(trafos.to_binary_kelp)
    ds.add_transform(trafos.to_tensor)


def apply_infer_trafos(ds: KelpTiledDataset) -> None:
    ds.add_transform(drop_channels)
    ds.add_transform(trafos.xr_to_np)
    ds.add_transform(trafos.channel_first)
    ds.add_transform(trafos.to_binary_kelp)
    ds.add_transform(trafos.to_binary_kelp)
    ds.add_transform(trafos.to_tensor)


def get_dataset(random_seed: int = 42):
    ds_kwargs = {
        "img_nc_path": "data_ncf/train_imgs_fe.nc",
        "mask_nc_path": "data_ncf/train_masks.ncf",
        "n_rand_tiles": 25,
        "tile_size": 64,
        "random_seed": random_seed
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


def get_loaders(random_seed: int = 42, **loader_kwargs):
    ds_train, ds_val, ds_test = get_dataset(random_seed=random_seed)

    # Shuffle data for training
    train_loader = torch.utils.data.DataLoader(ds_train, shuffle=True, **loader_kwargs)

    # Normal sampling for val and test
    val_loader = torch.utils.data.DataLoader(ds_val, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(ds_test, **loader_kwargs)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders(
        num_workers=32, batch_size=1024, random_seed=42
    )

    # Train
    model = LitBinaryClf(n_ch=8)
    trainer = L.Trainer(
        devices=1,
        max_epochs=5,
        log_every_n_steps=10,
        callbacks=[
            # EarlyStopping(monitor="val_dice", patience=3, mode="max"),
            LearningRateMonitor(logging_interval="epoch")
        ],
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # New trainer on just one device
    trainer = L.Trainer(devices=1)
    trainer.test(model, dataloaders=test_loader)
