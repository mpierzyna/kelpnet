"""
REMEMBER THAT MASK CAN BE `None`!
"""
import numba
import torch
import numpy as np
import cv2
import albumentations as A


def if_mask_valid(fn):
    """If mask is `None`, wrapped function is not applied and mask is returned as is."""
    def wrapper(mask, *args, **kwargs):
        if mask is None:
            return mask
        return fn(mask, *args, **kwargs)
    return wrapper


@numba.njit(cache=True)
def add_rs_indices(img, mask):
    """0: SWIR, 1: NIR, 2: R, 3: G, 4: B"""
    swir = img[:, :, 0]
    nir = img[:, :, 1]
    r = img[:, :, 2]
    g = img[:, :, 3]
    b = img[:, :, 4]

    ndwi_1 = (g - nir) / (g + nir)
    ndwi_2 = (nir - swir) / (nir + swir)
    ndvi = (nir - r) / (nir + r)
    
    # Preallocate new array with new channels (for numba)
    ni, nj, nch = img.shape
    img_new = np.zeros((ni, nj, nch+3), dtype=img.dtype)
    img_new[:, :, :nch] = img
    img_new[:, :, nch] = ndwi_1
    img_new[:, :, nch+1] = ndwi_2
    img_new[:, :, nch+2] = ndvi

    return img_new, mask


def downsample(img, mask):
    """OpenCV also requires channel to be last dimension (not needed if augmentations used)"""
    img = cv2.resize(img, (256, 256))
    mask = if_mask_valid(cv2.resize)(mask, (256, 256))
    return img, mask


aug_pipeline = A.Compose([
    A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=-1, p=0.5),  # -1 is nan effectively and will be ignored in loss
    A.RandomCrop(256, 256, p=1),  # always crop to have dataset of same size
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
