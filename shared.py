from typing import List, Optional, Tuple

import albumentations as A
import numpy as np
import torch

import trafos
from data import Channel as Ch
from data import KelpTiledDataset, KelpNCDataset, get_train_val_test_masks, RandomTileSampler, RegularTileSampler

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


def apply_train_trafos(ds: KelpTiledDataset, mode: str) -> None:
    aug_pipeline = A.Compose(
        [
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ]
    )

    def apply_aug(img, mask):
        if mode == "binary":
            res = aug_pipeline(image=img)
            return res["image"], mask
        else:
            res = aug_pipeline(image=img, mask=mask)
            return res["image"], res["mask"]

    if mode == "binary":
        ds.add_transform(trafos.to_binary_kelp)

    ds.add_transform(trafos.xr_to_np)
    ds.add_transform(apply_aug)  # Random augmentation only during training!
    ds.add_transform(trafos.channel_first)
    ds.add_transform(trafos.to_tensor)


def apply_infer_trafos(ds: KelpTiledDataset, mode: str) -> None:
    if mode == "binary":
        ds.add_transform(trafos.to_binary_kelp)

    ds.add_transform(trafos.xr_to_np)
    ds.add_transform(trafos.channel_first)
    ds.add_transform(trafos.to_tensor)


def get_dataset(use_channels: Optional[List[int]], split_seed: int, tile_seed: int, mode: str):
    """Get dataset with channel subset and split into train/val/test.
    Two seeds are used in an ensemble case:
    - `split_seed` should be the same for all members to get reproducible (independent) train/val/test splits.
    - `tile_seed` should be different for each member to get different random tiles for each member.
    """
    tile_size = 64
    ds_kwargs = {
        "img_nc_path": "data_ncf/train_imgs_fe.nc",
        "mask_nc_path": "data_ncf/train_masks.ncf",
        "use_channels": use_channels,
    }
    ds = KelpNCDataset(**ds_kwargs)  # Only load standard dataset for shape info for splitting

    # Split data into train/val/test
    mask_train, mask_val, mask_test = get_train_val_test_masks(len(ds.imgs), random_seed=split_seed)

    # Load dataset without outlier filter
    ds_train = KelpTiledDataset(**ds_kwargs, sample_mask=mask_train,
                                tile_sampler=RandomTileSampler(n_tiles=50, tile_size=tile_size, random_seed=tile_seed))
    apply_train_trafos(ds_train, mode=mode)

    ds_val = KelpTiledDataset(**ds_kwargs, sample_mask=mask_val,
                              tile_sampler=RegularTileSampler(tile_size=tile_size, overlap=16))
    ds_test = KelpTiledDataset(**ds_kwargs, sample_mask=mask_test,
                               tile_sampler=RegularTileSampler(tile_size=tile_size, overlap=16))
    apply_infer_trafos(ds_val, mode=mode)
    apply_infer_trafos(ds_test, mode=mode)

    return ds_train, ds_val, ds_test


def get_submission_dataset() -> KelpTiledDataset:
    tile_size = 64
    regular_ts = RegularTileSampler(tile_size=tile_size, overlap=16)
    ds = KelpTiledDataset(img_nc_path="data_ncf/test_imgs_fe.nc", mask_nc_path=None, tile_sampler=regular_ts)
    apply_infer_trafos(ds, mode="seg")
    return ds


def get_loaders(use_channels: Optional[List[int]], split_seed: int, tile_seed: int, mode: str, **loader_kwargs):
    ds_train, ds_val, ds_test = get_dataset(use_channels=use_channels, split_seed=split_seed, tile_seed=tile_seed, mode=mode)

    # Load data to RAM for fast training
    ds_train.load()
    if mode == "seg":
        ds_train.drop_no_kelp()
    ds_val.load()
    ds_test.load()

    # Shuffle data for training
    train_loader = torch.utils.data.DataLoader(ds_train, shuffle=True, **loader_kwargs)

    # Normal sampling for val and test
    val_loader = torch.utils.data.DataLoader(ds_val, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(ds_test, **loader_kwargs)

    return train_loader, val_loader, test_loader
