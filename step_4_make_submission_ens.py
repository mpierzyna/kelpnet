import datetime
import os
import pathlib

import joblib
import rasterio

from data import KelpNCDataset

SUBMISSION_ROOT = pathlib.Path("submission")
SUBMISSION_ROOT.mkdir(exist_ok=True)


def get_tiff_profile():
    """Return profile from sample tiff file for submission tiffs"""
    return rasterio.open("./data/sub_sample_kelp.tif").profile


if __name__ == "__main__":
    # Load data and predictions
    ds_sub = KelpNCDataset(img_nc_path="data_ncf/test_imgs_fe.nc", mask_nc_path=None)
    y_hat = joblib.load("pred_2staged_submission.joblib")

    # Upsample and save to tiff
    submission_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    submission_dir = SUBMISSION_ROOT / submission_name
    (submission_dir / "pred").mkdir(exist_ok=True, parents=True)  # Dir where predictions are saved

    # Tiff profile
    profile = get_tiff_profile()

    for y_hat_i, tile_id in zip(y_hat, ds_sub.tile_ids):
        # Write as tiff
        with rasterio.open(submission_dir / "pred" / f"{tile_id}_kelp.tif", "w", **profile) as dst:
            dst.write(y_hat_i, 1)

    # Tar it
    os.system(f"cd {submission_dir} && tar -czf {submission_name}_pred.tar.gz pred")
