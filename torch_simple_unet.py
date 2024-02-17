import pathlib
from typing import Optional

import click
import lightning as L
import lion_pytorch
import pytorch_toolbelt.losses
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor

import shared

torch.set_float32_matmul_precision("high")


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
        # self.e3 = EncoderBlock(128, 256)
        # self.e4 = EncoderBlock(256, 512)

        # Bottleneck
        self.b = ConvBlock(128, 256)

        # Decoder
        # self.d1 = DecoderBlock(1024, 512)
        # self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        # Classifier
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()  # kelp cover is a binary mask

    def forward(self, inputs):
        # Encoder
        s1, p = self.e1(inputs)
        s2, p = self.e2(p)
        # s3, p = self.e3(p)
        # s4, p = self.e4(p)

        # Bottleneck
        d = self.b(p)

        # Decoder
        # d = self.d1(b, s4)
        # d = self.d2(d, s3)
        d = self.d3(d, s2)
        d = self.d4(d, s1)

        # Classifier
        outputs = self.outputs(d)
        outputs = outputs.squeeze(1)  # remove channel dimension since we only have one channel
        return torch.sigmoid(outputs)


class LitUNet(L.LightningModule):
    def __init__(self, n_ch: int):
        """todo: change to take list of channels as input"""
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

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self.model(x)

    def configure_optimizers(self):
        optimizer = lion_pytorch.Lion(self.parameters(), lr=5e-4, weight_decay=1e-1)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.88)
        return [optimizer], [lr_scheduler]


def train(*, n_ch: Optional[int],  i_member: int, i_device: int, ens_dir: str):
    # Make sure ens_root exists and is empty
    ens_dir = pathlib.Path(ens_dir)
    ens_dir.mkdir(exist_ok=True)

    # Get random seed for this member
    random_seed = shared.get_local_seed(i_member)

    # Select channel subset
    use_channels, n_ch = shared.get_channel_subset(n_ch, random_seed)
    print(f"Member {i_member} uses channels: {use_channels}.")

    train_loader, val_loader, test_loader = shared.get_loaders(
        use_channels=use_channels, split_seed=shared.GLOBAL_SEED, tile_seed=random_seed, mode="seg",
        num_workers=0, batch_size=1024, pin_memory=True
    )

    # Save best models
    ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_dice",
        mode="max",
        dirpath=ens_dir,
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
    trainer = L.Trainer(devices=[i_device], logger=False)
    trainer.test(model, dataloaders=test_loader)


@click.command()
@click.option("--ens_dir", type=str, default="ens_seg/dev")
@click.argument("i_member", type=int)
@click.argument("i_device", type=int)
def train_cli(ens_dir: str, i_member: int, i_device: int):
    train(n_ch=3, i_member=i_member, i_device=i_device, ens_dir=ens_dir)


if __name__ == "__main__":
    train_cli()
