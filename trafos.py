"""
REMEMBER THAT MASK CAN BE `None`!
"""

import xarray as xr
import numba
import torch
import numpy as np
import cv2
import albumentations as A
import logging

from data import Channel as Ch
from utils import compute_scaled_fft2

logger = logging.getLogger("kelp")


def if_mask_valid(fn):
    """If mask is `None`, wrapped function is not applied and mask is returned as is."""

    def wrapper(mask, *args, **kwargs):
        if mask is None:
            return mask
        return fn(mask, *args, **kwargs)

    return wrapper


# @numba.njit(cache=True)
def add_rs_indices(img, mask):
    """0: SWIR, 1: NIR, 2: R, 3: G, 4: B"""
    swir = img[:, :, Ch.SWIR.value]
    nir = img[:, :, Ch.NIR.value]
    r = img[:, :, Ch.R.value]
    g = img[:, :, Ch.G.value]
    b = img[:, :, Ch.B.value]

    # I expect +- inf due to division by zero and fix them later
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi_1 = (g - nir) / (g + nir)
        ndwi_2 = (nir - swir) / (nir + swir)
        ndvi = (nir - r) / (nir + r)
        gndvi = (nir - g) / (nir + g)
        ndti = (r - g) / (r + g)
        evi = 2.5 * (nir - r) / (nir + 6 * r - 7.5 * b + 1)
        cari = ((nir - r) / (nir + r)) - ((nir - g) / (nir + g))

    # Preallocate new array with new channels (for numba)
    ni, nj, nch = img.shape
    img_new = np.zeros((ni, nj, nch + 4), dtype=img.dtype)
    img_new[:, :, : Ch.NDWI_1.value] = img
    img_new[:, :, Ch.NDWI_1.value] = ndwi_1
    img_new[:, :, Ch.NDWI_2.value] = ndwi_2
    img_new[:, :, Ch.NDVI.value] = ndvi
    img_new[:, :, Ch.GNDVI.value] = gndvi
    # img_new[:, :, Ch.NDTI.value] = ndti
    # img_new[:, :, Ch.EVI.value] = evi
    # img_new[:, :, Ch.CARI.value] = cari

    # If any of the rs indices caused inf due to division by zero fill with zero
    if (nan_count := np.isnan(img_new).sum() + np.isinf(img_new).sum()) > 0:
        logger.debug(f"Found {nan_count} NaN or inf values in rs indices. Filling with 0.")
        img_new = np.nan_to_num(img_new, nan=0, posinf=0, neginf=0)

    return img_new, mask


def xr_to_np(xr_img: xr.DataArray, xr_mask: xr.DataArray):
    """Convert xarray to numpy because KelpNCDataset returns native xarray Dataarrays"""
    img = xr_img.to_numpy()
    if xr_mask is None:
        mask = None
    else:
        mask = xr_mask.to_numpy()
    return img, mask


def downsample(img, mask):
    """OpenCV also requires channel to be last dimension (not needed if augmentations used)"""
    img = cv2.resize(img, (256, 256))
    mask = if_mask_valid(cv2.resize)(mask, (256, 256))
    return img, mask


def upsample_to_512(img, mask):
    """OpenCV also requires channel to be last dimension (not needed if augmentations used)"""
    img = cv2.resize(img, (512, 512))
    mask = if_mask_valid(cv2.resize)(mask, (512, 512))
    return img, mask


aug_pipeline = A.Compose(
    [
        A.Rotate(
            border_mode=cv2.BORDER_CONSTANT, value=-1, p=0.5
        ),  # -1 is nan effectively and will be ignored in loss
        A.RandomCrop(256, 256, p=1),  # always crop to have dataset of same size
        A.HorizontalFlip(),
    ]
)


def augment(img, mask):
    """Apply augmentation pipeline. Expects opencv style channels, ie last"""
    res = aug_pipeline(image=img, mask=mask)
    return res["image"], res["mask"]


def channel_first(img, mask):
    """Torch wants channels first. Apply last!"""
    return np.rollaxis(img, 2, 0), mask


def to_tensor(img, mask):
    if mask is None:
        mask = torch.tensor([0])  # Cannot be None because DataLoader cannot concat None to batches
    else:
        mask = torch.tensor(mask, dtype=torch.uint8)
    return torch.tensor(img, dtype=torch.float32), mask


def add_fft2_ch(img, mask):
    nir_ft = compute_scaled_fft2(img[:, :, Ch.NIR.value]) / 20
    swir_ft = compute_scaled_fft2(img[:, :, Ch.SWIR.value]) / 20
    ndwi_1_ft = compute_scaled_fft2(img[:, :, Ch.NDWI_1.value]) / 20
    ndvi_ft = compute_scaled_fft2(img[:, :, Ch.NDVI.value]) / 20

    # Stack new channels
    img = np.dstack((img, nir_ft, swir_ft, ndwi_1_ft, ndvi_ft))

    return img, mask


if __name__ == "__main__":
    # Test if add_rs compiles correctly
    for _ in range(10):
        img = np.random.uniform(0, 1, size=(256, 256, 8))
        mask = np.random.randint(0, 1, size=(256, 256))  # just passed through anyway
        add_rs_indices(img, mask)
