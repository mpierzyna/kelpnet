import torch
import numpy as np
import cv2


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
    mask = cv2.resize(mask, (256, 256))
    return img, mask


def channel_first(img, mask):
    """Torch wants channels first. Apply last!"""
    return np.rollaxis(img, 2, 0), mask


def to_tensor(img, mask):
    return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.uint8)