import albumentations as A
import lightning as L
import torch
from torch import nn
import torch.utils.data
import rasterio
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from torchvision.models.segmentation import deeplabv3_resnet101
import pytorch_toolbelt.losses
import lion_pytorch
import torchmetrics
import warnings
import pandas as pd
import logging
from torch.nn import functional as F


from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.swin_transformer import _swin_transformer, SwinTransformerBlockV2, PatchMergingV2, SwinTransformer

from data import KelpDataset, split_train_test_val2
from data import Channel as Ch
import trafos


torch.set_float32_matmul_precision("high")
# Reading dataset raises warnings, which are anoying
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Setup logging
logger = logging.getLogger("kelp")
logging.basicConfig(level=logging.INFO)
logging.getLogger("rasterio").setLevel(logging.ERROR)


def drop_channels(img, mask):
    """Drop channels with low importance from img.
    Expects channels in last dimension.
    """
    _, _, n_ch = img.shape
    to_drop = [Ch.R, Ch.G, Ch.B, Ch.IS_CLOUD]
    to_keep = [ch for ch in range(n_ch) if ch not in to_drop]
    img = img[:, :, to_keep]
    return img, mask


def get_mod_swin(progress: bool = True) -> SwinTransformer:
    # Reduced depth so that compressed space is bigger
    return _swin_transformer(
        weights=None,
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18],
        # depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16],
        # num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        stochastic_depth_prob=0.5,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
    )


class SwinDeepLab(nn.Module):
    def __init__(self, n_ch: int):
        super().__init__()

        # Load swin and modify first layer to accept variable number of channels
        # https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py#L508
        swin = get_mod_swin()
        swin_init_conv = swin.features[0][0]
        swin.features[0][0] = nn.Conv2d(
            n_ch,
            out_channels=swin_init_conv.out_channels,
            kernel_size=swin_init_conv.kernel_size,
            stride=swin_init_conv.stride,
        )

        # Delete original swin head because we use our own segmentation head
        n_features = swin.head.in_features
        self.head = None

        self.swin = swin

        # DeepLabV3 head (https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py#L116)
        self.segmentation_head = DeepLabHead(n_features, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]

        # Use part of Swin's forward pass to extract features
        x = self.swin.features(x)
        x = self.swin.norm(x)
        x = self.swin.permute(x)

        # Decode with DeepLabV3 head
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        # remove channel dimension since we only have one channel
        x = x.squeeze(1)

        # Apply sigmoid activation for binary classification
        return torch.sigmoid(x)


class LitSwinDeepLab(L.LightningModule):
    def __init__(self, n_ch: int, ens_prediction: bool):
        super().__init__()
        self.model = SwinDeepLab(n_ch=n_ch)
        self.crit = pytorch_toolbelt.losses.DiceLoss(mode="binary", from_logits=False)

        # Additional metrics
        self.dice = torchmetrics.Dice()

        # Store arguments
        self.ens_prediction = ens_prediction
        self.n_ch = n_ch

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
        self.log(f"{prefix}_dice", self.dice(y_hat, y.int()), sync_dist=True)

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
        optimizer = lion_pytorch.Lion(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [lr_scheduler]


def apply_train_trafos(ds: KelpDataset) -> None:
    aug_pipeline = A.Compose(
        [
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ]
    )

    def apply_aug(img, mask):
        res = aug_pipeline(image=img, mask=mask)
        return res["image"], res["mask"]

    ds.add_transform(trafos.add_rs_indices)
    # ds.add_transform(trafos.add_fft2_ch)
    ds.add_transform(drop_channels)
    ds.add_transform(apply_aug)  # Random augmentation only during training!
    ds.add_transform(trafos.channel_first)
    ds.add_transform(trafos.to_tensor)


def apply_infer_trafos(ds: KelpDataset) -> None:
    ds.add_transform(trafos.add_rs_indices)
    # ds.add_transform(trafos.add_fft2_ch)
    ds.add_transform(drop_channels)
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
    train_loader, val_loader, test_loader = get_loaders(num_workers=8, batch_size=32, kf_weighing=False)

    # Train
    model = LitSwinDeepLab(n_ch=9, ens_prediction=True)
    print(model)
    trainer = L.Trainer(
        devices=1,
        max_epochs=20,
        log_every_n_steps=10,
        callbacks=[
            # EarlyStopping(monitor="val_dice", patience=3),
            LearningRateMonitor(logging_interval="epoch")
        ],
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # New trainer on just one device
    trainer.test(ckpt_path="best", dataloaders=test_loader)
