import cv2
import torch.utils.data
import lightning as L
import rasterio
import pathlib
import datetime

from data import KelpDataset
from torch_unet import  LitUNet

SUBMISSION_ROOT = pathlib.Path("submission")
SUBMISSION_ROOT.mkdir(exist_ok=True)


def upsample_mask(mask):
    """Upsample prediction again for submission"""
    mask = cv2.resize(mask, (350, 350))
    return mask


def make_binary_mask(mask):
    """Make binary mask from logits"""
    mask = mask > 0
    return mask


if __name__ == "__main__":
    # Load model
    unet = LitUNet.load_from_checkpoint("lightning_logs/version_9/checkpoints/epoch=24-step=6400.ckpt", n_ch=11)

    # Load data
    ds_sub = KelpDataset(img_dir="data/test_satellite/", mask_dir=None)
    unet.apply_test_trafos(ds_sub)
    ds_sub_loader = torch.utils.data.DataLoader(ds_sub, batch_size=16, num_workers=4)

    # Make predictions
    trainer = L.Trainer()
    y_hat = trainer.predict(unet, ds_sub_loader)  # Batch-wise prediction (list of tensors of dim [batch_size, 256, 256])
    y_hat = torch.concat(y_hat)  # Concatenate to single tensor of dim [n_samples, 256, 256]

    # Upsample and save to tiff
    submission_dir = SUBMISSION_ROOT / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    submission_dir.mkdir()

    for mask_hat, tile_id in zip(y_hat, ds_sub.tile_ids):
        mask_hat = upsample_mask(mask_hat.numpy())
        with rasterio.open(submission_dir / f"{tile_id}.tif", "w") as dst:
            dst.write(mask_hat, 1)
