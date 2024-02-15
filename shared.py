from typing import List, Optional, Tuple

import albumentations as A
import click
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
import pytorch_toolbelt.losses
import torchmetrics
import lion_pytorch
import pathlib

import trafos
from data import Channel as Ch
from data import KelpTiledDataset, get_train_val_test_masks

torch.set_float32_matmul_precision("high")


GLOBAL_SEED = 42  # DO NOT CHANGE

VALID_CHANNELS = [
    Ch.SWIR,
    Ch.NIR,
    Ch.R,
    Ch.G,
    Ch.B,
    Ch.NDWI_1,
    Ch.NDWI_2,
    Ch.NDVI,
    Ch.GNDVI,
    Ch.NDTI,
    Ch.EVI,
    Ch.CARI,
]


def get_local_seed(i_member: int) -> int:
    """Get reproducible random seed for member from global seed"""
    # Init rng with global seed to get rng for this member
    rng = np.random.default_rng(GLOBAL_SEED)
    random_seed = None
    for _ in range(i_member + 1):
        random_seed = rng.integers(0, 2**32 - 1)

    return random_seed


def get_channel_subset(n_ch: Optional[int], random_seed: int) -> Tuple[np.ndarray, int]:
    rng = np.random.default_rng(random_seed)
    if n_ch is None:
        # Default
        use_channels = [0, 1, 2, 6, 8, 9, 10, 11]
    else:
        # Random choice
        use_channels = rng.choice(VALID_CHANNELS, size=n_ch, replace=False)

    use_channels = np.array(use_channels)
    n_ch = len(use_channels)
    return use_channels, n_ch
