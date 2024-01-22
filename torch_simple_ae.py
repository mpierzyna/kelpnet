import pandas as pd

import lightning as L
from lion_pytorch import Lion

import torch
from torch.utils.data.dataset import random_split
from torch import nn

from data import KelpDataset
import trafos


class LitAutoEncoder(L.LightningModule):
    def __init__(self, img_ch):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_ch, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, img_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=1e-3)
        optimizer = Lion(self.parameters(), lr=1e-4, weight_decay=1e-2)
        return optimizer


if __name__ == "__main__":
    # Load data and add trafos
    quality_df = pd.read_csv("quality.csv")
    ds = KelpDataset(img_dir="data/train_satellite/", mask_dir="data/train_kelp/", dir_mask=quality_df["nan_fraction"] == 0)
    ds.add_transform(trafos.add_rs_indices)
    ds.add_transform(trafos.downsample)
    ds.add_transform(trafos.channel_first)
    ds.add_transform(trafos.to_tensor)

    # Split data
    ds_train, ds_val, ds_test = torch.utils.data.random_split(ds, [.7, .15, .15])

    # Get shape
    X, y = ds_train[0]
    img_ch, img_h, img_w = X.shape

    # Start training
    autoencoder = LitAutoEncoder(img_ch=img_ch)
    train_loader = torch.utils.data.DataLoader(ds_train, num_workers=4)
    val_loader = torch.utils.data.DataLoader(ds_val, num_workers=4)

    trainer = L.Trainer(limit_train_batches=256, max_epochs=25)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader)
