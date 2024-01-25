import numpy as np
import cv2
import torch.utils.data
import lightning as L
import rasterio
import pathlib
import datetime
import os

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

def get_tiff_profile():
    """Return profile from sample tiff file for submission tiffs"""
    return rasterio.open("./data/sub_sample_kelp.tif").profile


if __name__ == "__main__":
    # Load model
    unet = LitUNet.load_from_checkpoint("lightning_logs/version_46/checkpoints/epoch=16-step=595.ckpt", n_ch=11)

    # Load data
    ds_sub = KelpDataset(img_dir="data/test_satellite/", mask_dir=None)
    unet.apply_test_trafos(ds_sub)
    ds_sub_loader = torch.utils.data.DataLoader(ds_sub, batch_size=16, num_workers=4)

    # Make predictions
    trainer = L.Trainer(devices=1)
    y_hat = trainer.predict(unet, ds_sub_loader)  # Batch-wise prediction (list of tensors of dim [batch_size, 256, 256])
    y_hat = torch.concat(y_hat)  # Concatenate to single tensor of dim [n_samples, 256, 256]
    y_hat = y_hat > 0.5  # Make binary mask

    # Upsample and save to tiff
    submission_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    submission_dir = SUBMISSION_ROOT / submission_name
    submission_dir.mkdir(exist_ok=True, parents=True)

    # Tiff profile
    profile = get_tiff_profile()

    for y_hat_i, tile_id in zip(y_hat, ds_sub.tile_ids):
        # Upsample
        y_hat_i = y_hat_i.numpy().astype(np.uint8)
        y_hat_i = upsample_mask(y_hat_i)

        # Write as tiff
        with rasterio.open(submission_dir / f"{tile_id}_kelp.tif", "w", **profile) as dst:
            dst.write(y_hat_i, 1)

    # Tar it
    os.system(f"cd {SUBMISSION_ROOT} && tar -czf {submission_name}.tar.gz {submission_name}")
