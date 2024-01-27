import torch
from typing import Optional
import pathlib
import numpy as np
import rasterio
import pandas as pd

from torch.utils.data import Dataset


def load_img(fpath_img: str, fpath_mask: Optional[str]):
    """Read image from file and return as float32 tensor"""
    # Load image
    img = rasterio.open(fpath_img).read().astype(np.float32)
    img = np.rollaxis(img, 0, 3)  # tf expects channels in last dimension
    img[:, :, :5] = img[:, :, :5] / 65536.  # scale bands to [0, 1]

    # I assume, detection only possible when not cloudy and not land
    is_cloud = img[:, :, 5].astype(bool)
    is_land = img[:, :, 6] > 0  # make land masked based on DEM
    not_cloud_land = (~is_cloud) & (~is_land)
    img = np.concatenate([img, not_cloud_land[:, :, None]], axis=2)
    img[:, :, 6] = is_land

    # Load mask (if present)
    if fpath_mask is None:
        mask = None
    else:
        mask = rasterio.open(fpath_mask).read()
        mask = mask[0].astype(np.uint8)

    return img, mask


class KelpDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: Optional[str], dir_mask: Optional[np.ndarray] = None):
        # List images (X)
        img_list = sorted(pathlib.Path(img_dir).glob("*.tif"))

        if mask_dir is None:
            # No masks, e.g. for test set
            mask_list = [None] * len(img_list)
        else:
            # List masks (y)
            mask_list = sorted(pathlib.Path(mask_dir).glob("*.tif"))

            # Sanity check that images and labels are associated correctly
            for img_path, mask_path in zip(img_list, mask_list):
                img_tile_id, _ = img_path.stem.split("_")
                mask_tile_id, _ = mask_path.stem.split("_")
                assert img_tile_id == mask_tile_id

        # Apply directory mask to image and mask
        self.img_list = np.array(img_list)
        self.mask_list = np.array(mask_list)
        if dir_mask is not None:
            self.img_list = self.img_list[dir_mask]
            self.mask_list = self.mask_list[dir_mask]

        # Save tile ids
        self.tile_ids = [p.stem.split("_")[0] for p in self.img_list]

        # Store trafos
        self.transforms = []

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, mask = load_img(self.img_list[idx], self.mask_list[idx])
        for tf in self.transforms:
            img, mask = tf(img, mask)
        return img, mask

    def add_transform(self, tf):
        self.transforms.append(tf)


class MultiTaskKelpDataset(KelpDataset):
    def __init__(self, img_dir: str, mask_dir: Optional[str], cog_path: Optional[str],
                 dir_mask: Optional[np.ndarray] = None):
        super().__init__(img_dir, mask_dir, dir_mask)

        if cog_path is None:
            self.df_cog = None
        else:
            df_cog = pd.read_csv(cog_path, index_col=0)
            if dir_mask is not None:
                df_cog = df_cog.loc[dir_mask].reset_index(drop=True)
            self.df_cog = df_cog

        self.transforms_cog = []

    def __getitem__(self, idx):
        img, mask = super().__getitem__(idx)
        if self.df_cog is None:
            return img, (mask, None)

        cog = self.df_cog.iloc[idx].values.astype(np.float32)
        for tf in self.transforms_cog:
            cog = tf(cog)
        return img, (mask, cog)

    def add_regr_transform(self, tf):
        self.transforms_cog.append(tf)


def split_train_test_val(ds: KelpDataset, seed=42):
    """Split data into train (70%), validation (15%) and test (15%) set."""
    gen = torch.Generator().manual_seed(seed)
    ds_train, ds_val, ds_test = torch.utils.data.random_split(ds, [.7, .15, .15], generator=gen)
    return ds_train, ds_val, ds_test
