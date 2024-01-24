"""
REMEMBER THAT MASK CAN BE `None`!
"""
import torch
import numpy as np
import cv2


def if_mask_valid(fn):
    """If mask is `None`, wrapped function is not applied and mask is returned as is."""
    def wrapper(mask, *args, **kwargs):
        if mask is None:
            return mask
        return fn(mask, *args, **kwargs)
    return wrapper


def add_rs_indices(img, mask):
    """0: SWIR, 1: NIR, 2: R, 3: G, 4: B"""
    swir = img[:, :, 0][:, :, None]
    nir = img[:, :, 1][:, :, None]
    r = img[:, :, 2][:, :, None]
    g = img[:, :, 3][:, :, None]
    b = img[:, :, 4][:, :, None]

    ndwi_1 = (g - nir) / (g + nir)
    ndwi_2 = (nir - swir) / (nir + swir)
    ndvi = (nir - r) / (nir + r)

    img = np.concatenate([img, ndwi_1, ndwi_2, ndvi], axis=2)
    return img, mask


def downsample(img, mask):
    """OpenCV also requires channel to be last dimension"""
    img = cv2.resize(img, (256, 256))
    mask = if_mask_valid(cv2.resize)(mask, (256, 256))
    return img, mask


def random_flip(img, mask):
    """Randomly flip image and mask horizontally or vertically."""
    a, b = np.random.random(size=2)
    if a > .5:
        if b > .5:
            return cv2.flip(img, 0), if_mask_valid(cv2.flip)(mask, 0)
        return cv2.flip(img, 1), if_mask_valid(cv2.flip)(mask, 1)
    return img, mask


def channel_first(img, mask):
    """Torch wants channels first. Apply last!"""
    return np.rollaxis(img, 2, 0), mask


def to_tensor(img, mask):
    if mask is None:
        mask = torch.tensor([0])  # Cannot be None because DataLoader cannot concat None to batches
    else:
        mask = torch.tensor(mask, dtype=torch.uint8)
    return torch.tensor(img, dtype=torch.float32), mask
