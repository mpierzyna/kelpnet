import pandas as pd
import rasterio

import torch
from torch import nn

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lion_pytorch import Lion
import torchmetrics

from data import KelpDataset
import trafos
import warnings

# Reading dataset raises warnings, which are anoying
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


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
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
         
        # Bottleneck
        b = self.b(p4)
        
        # Decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
         
        # Classifier
        outputs = self.outputs(d4)
        # outputs = self.sigmoid(outputs)
        outputs = outputs.squeeze(1)  # remove channel dimension since we only have one channel
        return outputs


class LitUNet(L.LightningModule):
    def __init__(self, n_ch):
        super().__init__()
        self.model = UNet(n_ch=n_ch)
        self.dice = torchmetrics.Dice()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_dice", self.dice(y_hat, y.int()))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_dice", self.dice(y_hat, y.int()), sync_dist=True)

    def configure_optimizers(self):
        optimizer = Lion(self.parameters(), lr=1e-3, weight_decay=1e-2)
        return optimizer


def get_dataset():
    # Load data and add trafos
    quality_df = pd.read_csv("quality.csv")
    ds = KelpDataset(img_dir="data/train_satellite/", mask_dir="data/train_kelp/", dir_mask=quality_df["nan_fraction"] == 0)
    ds.add_transform(trafos.add_rs_indices)
    ds.add_transform(trafos.downsample)
    ds.add_transform(trafos.channel_first)
    ds.add_transform(trafos.to_tensor)
    ds.add_transform(lambda X, y: (X, y.float()))  # y is float for BCELoss

    # Split data
    gen = torch.Generator().manual_seed(42)
    ds_train, ds_val, ds_test = torch.utils.data.random_split(ds, [.7, .15, .15], generator=gen)
    return ds_train, ds_val, ds_test


if __name__ == "__main__":
    ds_train, ds_val, _ = get_dataset()
    train_loader = torch.utils.data.DataLoader(ds_train, num_workers=4)
    val_loader = torch.utils.data.DataLoader(ds_val, num_workers=4)

    # Train
    model = LitUNet(n_ch=11)
    trainer = L.Trainer(
        limit_train_batches=256, 
        max_epochs=25,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3)]
    )
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader,
    )
