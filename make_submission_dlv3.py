import datetime
import os
import pathlib
import shutil

import cv2
import lightning as L
import numpy as np
import rasterio
import torch.utils.data

from data import KelpDataset
import torch_deeplabv3 as dlv3

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
    ckpt_path = pathlib.Path("lightning_logs/version_14/checkpoints/epoch=11-step=1488.ckpt")
    model = dlv3.LitDeepLabV3.load_from_checkpoint(ckpt_path, n_ch=11, ens_prediction=True)

    # Load data (INPAINTED!)
    ds_sub = KelpDataset(img_dir="data_inpainted/test_satellite/", mask_dir=None)
    dlv3.apply_infer_trafos(ds_sub)
    ds_sub_loader = torch.utils.data.DataLoader(ds_sub, batch_size=16, num_workers=4)

    # Make predictions (already returned as binary mask)
    trainer = L.Trainer(devices=1, logger=False)
    y_hat = trainer.predict(model, ds_sub_loader)
    y_hat = torch.concat(y_hat)  # Concatenate to single tensor

    # Upsample and save to tiff
    submission_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    submission_dir = SUBMISSION_ROOT / submission_name
    (submission_dir / "pred").mkdir(exist_ok=True, parents=True)  # Dir where predictions are saved

    # Tiff profile
    profile = get_tiff_profile()

    for y_hat_i, tile_id in zip(y_hat, ds_sub.tile_ids):
        # Upsample
        y_hat_i = y_hat_i.numpy().astype(np.uint8)
        y_hat_i = upsample_mask(y_hat_i)

        # Write as tiff
        with rasterio.open(submission_dir / "pred" / f"{tile_id}_kelp.tif", "w", **profile) as dst:
            dst.write(y_hat_i, 1)

    # Tar it
    os.system(f"cd {submission_dir} && tar -czf {submission_name}_pred.tar.gz pred")

    # Backup model
    shutil.copytree(ckpt_path.parent.parent, submission_dir / "model")
