import pathlib
from typing import Optional

import click
import lightning as L
import lion_pytorch
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor

import shared


class BinaryClfCNN(nn.Module):
    def __init__(self, n_ch: int, fc_size: int, p_dropout: float):
        res_in = 64
        ch1, ch2, ch3 = 64, 128, 256

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
            nn.Conv2d(in_channels=ch2, out_channels=ch3, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=p_dropout),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(ch3 * (res_in // 8) * (res_in // 8), fc_size),
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
        y_hat = torch.sigmoid(y_hat)
        return y_hat

    def configure_optimizers(self):
        optimizer = lion_pytorch.Lion(self.parameters(), lr=5e-4, weight_decay=1e-1)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.88)
        return [optimizer], [lr_scheduler]


def train(*, n_ch: Optional[int],  i_member: int, i_device: int, ens_dir: str):
    # Make sure ens_root exists and is empty
    ens_dir = pathlib.Path(ens_dir)
    ens_dir.mkdir(exist_ok=True, parents=True)

    # Get random seed for this member
    random_seed = shared.get_local_seed(i_member)

    # Select channel subset
    use_channels, n_ch = shared.get_channel_subset(n_ch, random_seed)
    print(f"Member {i_member} uses channels: {use_channels}.")

    train_loader, val_loader, test_loader = shared.get_loaders(
        use_channels=use_channels, split_seed=shared.GLOBAL_SEED, tile_seed=random_seed, mode="binary",
        num_workers=0, batch_size=1024, pin_memory=True
    )

    # Save best models
    ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_dice",
        mode="max",
        dirpath=ens_dir,
        filename=f"clf_{i_member}_" + "-".join(use_channels.astype(str)) + "_{epoch:02d}_{val_dice:.2f}",
    )

    # Train
    model = LitBinaryClf(n_ch=n_ch)
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
@click.option("--ens_dir", type=str, default="ens_clf/dev")
@click.argument("i_member", type=int)
@click.argument("i_device", type=int)
def train_cli(ens_dir: str, i_member: int, i_device: int):
    train(n_ch=3, i_member=i_member, i_device=i_device, ens_dir=ens_dir)


if __name__ == "__main__":
    train_cli()
    # train(n_ch=3, i_member=0, i_device=0, ens_root="ens_clf/dev")
    # train_cli([0, 0])
