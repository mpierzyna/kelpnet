"""
REMEMBER THAT MASK CAN BE `None`!
"""
import numba
import torch
import numpy as np
import cv2
import albumentations as A

from data import Channel as Ch

NAN_INT = -10  # int representing NaN


def if_mask_valid(fn):
    """If mask is `None`, wrapped function is not applied and mask is returned as is."""
    def wrapper(mask, *args, **kwargs):
        if mask is None:
            return mask
        return fn(mask, *args, **kwargs)
    return wrapper


@numba.njit(cache=True)
def add_rs_indices(img, mask):
    """ Add remote sensing indices as additional features
    Default channels: 0: SWIR, 1: NIR, 2: R, 3: G, 4: B
    Masks: 5: is_cloud, 6: is_land, 7: not_cloud_land

    Receives images with NaN on purpose, so that computed indices are also NaN where they should be.
    """
    swir = img[:, :, Ch.SWIR.value]
    nir = img[:, :, Ch.NIR.value]
    r = img[:, :, Ch.R.value]
    g = img[:, :, Ch.G.value]
    b = img[:, :, Ch.B.value]

    ndwi_1 = (g - nir) / (g + nir)
    ndwi_2 = (nir - swir) / (nir + swir)
    ndvi = (nir - r) / (nir + r)
    
    # Preallocate new array with new channels (for numba)
    ni, nj, nch = img.shape
    img_new = np.zeros((ni, nj, nch+3), dtype=img.dtype)
    img_new[:, :, :Ch.NDWI_1.value] = img
    img_new[:, :, Ch.NDWI_1.value] = ndwi_1
    img_new[:, :, Ch.NDWI_2.value] = ndwi_2
    img_new[:, :, Ch.NDVI.value] = ndvi

    return img_new, mask


def center_channels(img, mask):
    """Center image channels around global mean (if band or index) or 0 for masks."""
    # Channels containing masks are centered around 0
    ch_masks = [Ch.IS_CLOUD, Ch.IS_LAND, Ch.NOT_CLOUD_LAND]
    img[:, :, ch_masks] = img[:, :, ch_masks] - 0.5

    # Channels containing bands or indices are centered around precomputed global mean
    img_global_mean = np.array([ 0.1536,  0.1645,  0.1337,  0.1355,  0.1307, -0.0678,  0.0258,  0.0761])
    ch_not_masks = [c for c in Ch if c not in ch_masks]
    img[:, :, ch_not_masks] = img[:, :, ch_not_masks] - img_global_mean[None, None, :]
    return img, mask


def fill_nans(img, mask):
    """Fill NaNs with -1"""
    img = np.nan_to_num(img, nan=NAN_INT)   # NaNs are set to very low number
    return img, mask


def downsample(img, mask):
    """OpenCV also requires channel to be last dimension (not needed if augmentations used)"""
    img = cv2.resize(img, (256, 256))
    mask = if_mask_valid(cv2.resize)(mask, (256, 256))
    return img, mask


aug_pipeline = A.Compose([
    A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=np.nan, p=0.5),  # NaN is set to -1 later and ignored in loss
    # A.RandomCrop(256, 256, p=1),  # always crop to have dataset of same size
    A.HorizontalFlip(),
])


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


if __name__ == "__main__":
    # Test if add_rs compiles correctly
    for _ in range(10):
        img = np.random.uniform(0, 1, size=(256, 256, 8))
        img = np.where(img < 0.2, np.nan, img)  # set small part to NaN to check if numba handels it correctly
        mask = np.random.randint(0, 1, size=(256, 256))  # just passed through anyway
        add_rs_indices(img, mask)
        assert np.any(np.isnan(img))  # NaNs still there?
