import enum
import torch
from typing import Optional, List, Tuple, Self
import abc
import pathlib
import numpy as np
import rasterio
import pandas as pd
import logging
import xarray as xr

from torch.utils.data import Dataset


logger = logging.getLogger("kelp")


class Channel(enum.IntEnum):
    # Default channels
    SWIR = 0
    NIR = 1
    R = 2
    G = 3
    B = 4
    IS_CLOUD = 5
    IS_LAND = 6  # elevation height originally
    NOT_CLOUD_LAND = 7  # Extra mask
    # RS indices
    NDWI_1 = 8
    NDWI_2 = 9
    NDVI = 10
    GNDVI = 11
    NDTI = 12
    EVI = 13
    CARI = 14


def load_img(fpath_img: str, fpath_mask: Optional[str]):
    """Read image from file and return as float32 tensor"""
    # Load image
    logger.debug(f"Loading {fpath_img} and {fpath_mask}")
    img = rasterio.open(fpath_img).read().astype(np.float32)
    img = np.rollaxis(img, 0, 3)  # tf expects channels in last dimension
    img[:, :, :5] = img[:, :, :5] / 65536.  # scale bands to [0, 1]

    # I assume, detection only possible when not cloudy and not land
    is_cloud = img[:, :, Channel.IS_CLOUD].astype(bool)
    is_land = img[:, :, Channel.IS_LAND] > 0  # make land masked based on DEM
    not_cloud_land = (~is_cloud) & (~is_land)
    img = np.concatenate([img, not_cloud_land[:, :, None]], axis=2)
    img[:, :, Channel.IS_LAND] = is_land

    # For debugging
    if np.any(img < 0):
        raise ValueError(f"Negative values in {fpath_img}")

    # Load mask (if present)
    if fpath_mask is None:
        mask = None
    else:
        mask = rasterio.open(fpath_mask).read()
        mask = mask[0].astype(np.uint8)

    return img, mask


class KelpDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: Optional[str], dir_mask: Optional[np.ndarray] = None):
        # Store input params
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.dir_mask = dir_mask

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
            logger.debug(f"Applying {tf.__name__} to {idx}")
            img, mask = tf(img, mask)
            if np.isnan(img).sum() > 0:
                logger.warning(f"NaNs after {tf.__name__} in {idx}")
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


class KelpNCDataset(Dataset):
    def __init__(self, img_nc_path: str, mask_nc_path: Optional[str] = None, sample_mask: Optional[np.ndarray] = None, use_channels: Optional[List[int]] = None):
        # Open netcdf datasets where one sample with all channels is one chunk
        chunks = {"sample": 1, "i": None, "j": None, "ch": None}

        self.imgs = xr.open_dataarray(img_nc_path, engine="netcdf4", chunks=chunks)
        if sample_mask is not None:
            self.imgs = self.imgs.sel(sample=sample_mask)

        self.masks = None
        if mask_nc_path is not None:
            self.masks = xr.open_dataarray(mask_nc_path, engine="netcdf4", chunks=chunks)
            if sample_mask is not None:
                self.masks = self.masks.sel(sample=sample_mask)

        # For compatibility with KelpDataset
        self.tile_ids = self.imgs["sample"].values

        # Store inputs
        self.img_nc_path = img_nc_path
        self.mask_nc_path = mask_nc_path
        self.sample_mask = sample_mask
        self.use_channels = use_channels

        # Store trafos
        self.transforms = []

        self.loaded = False

    def __len__(self):
        return len(self.imgs)

    def _apply_trafos(self, idx, img, mask):
        for tf in self.transforms:
            logger.debug(f"Applying {tf.__name__} to {idx}")
            img, mask = tf(img, mask)
            if np.isnan(img).sum() > 0:
                logger.warning(f"NaNs after {tf.__name__} in {idx}")

        return img, mask

    def __getitem__(self, idx):
        img = self.imgs.isel(sample=idx)
        if self.use_channels is not None:
            img = img.isel(ch=self.use_channels)

        if self.masks is None:
            mask = None
        else:
            mask = self.masks.isel(sample=idx)

        img, mask = self._apply_trafos(idx, img, mask)
        return img, mask

    def add_transform(self, tf):
        self.transforms.append(tf)

    def load(self):
        """Load all data into memory."""
        if not self.loaded:
            print("Loading data into memory...")
            self.imgs.load()
            if self.masks is not None:
                self.masks.load()
            self.loaded = True


class BaseTileSampler(abc.ABC):
    def __init__(self, n_tiles: int, tile_size: int):
        self.n_tiles = n_tiles
        self.tile_size = tile_size

    def fit(self, imgs: xr.Dataset) -> Self:
        self.n_imgs_ = len(imgs)
        self.orig_size_ = imgs.sizes["i"]  # assume quadratic images

    def get_img_inds(self) -> np.ndarray:
        """Repeat index of each image n_tiles times for each tile."""
        return np.repeat(np.arange(self.n_imgs_), self.n_tiles)

    @abc.abstractmethod
    def get_img_tile_inds(self):
        """Return list of indices for tiles to sample from each image."""


class RandomTileSampler(BaseTileSampler):
    def __init__(self, n_tiles: int, tile_size: int, random_seed: int):
        super().__init__(n_tiles, tile_size)
        self.random_seed = random_seed

    def get_img_tile_inds(self):
        """Get `n_tiles` random tile indices per image based on `tile_size` and `orig_size_` of images"""
        rng = np.random.default_rng(seed=self.random_seed)
        return rng.integers(0, self.orig_size_ - self.tile_size + 1, size=(self.n_imgs_ * self.n_tiles, 2))


class KelpTiledDataset(KelpNCDataset):
    def __init__(self, img_nc_path: str, tile_sampler: BaseTileSampler, mask_nc_path: Optional[str] = None,
                 sample_mask: Optional[np.ndarray] = None, use_channels: Optional[List[int]] = None):
        # Execute parent code for loading
        super().__init__(img_nc_path, mask_nc_path, sample_mask, use_channels=use_channels)

        # Get tiles from tile sampler
        tile_sampler.fit(self.imgs)
        self.img_inds = tile_sampler.get_img_inds()
        self.img_tile_inds = tile_sampler.get_img_tile_inds()
        self.tile_sampler = tile_sampler

    def __len__(self):
        return len(self.img_inds)

    def __getitem__(self, idx):
        idx_img = self.img_inds[idx]
        tile_i, tile_j = self.img_tile_inds[idx]
        tilesize = self.tile_sampler.tile_size

        img = self.imgs.isel(sample=idx_img)
        img = img.sel(i=slice(tile_i, tile_i + tilesize), j=slice(tile_j, tile_j + tilesize))
        if self.use_channels is not None:
            img = img.isel(ch=self.use_channels)

        if self.masks is None:
            mask = None
        else:
            mask = self.masks.isel(sample=idx_img)
            mask = mask.sel(i=slice(tile_i, tile_i + tilesize), j=slice(tile_j, tile_j + tilesize))

        img, mask = self._apply_trafos(idx, img, mask)
        return img, mask

    @classmethod
    def only_with_kelp(cls, **kwargs):
        """Create a new KelpTiledDataset with only samples that contain kelp."""
        # Load dataset
        ds = cls(**kwargs)

        # Get kelp mask
        has_kelp = ds.masks.sum(["i", "j"]) > 0

        # Create new dataset with only kelp samples
        return cls(**kwargs, sample_mask=has_kelp)


def split_train_test_val(ds: KelpDataset, seed=42):
    """Split data into train (70%), validation (15%) and test (15%) set."""
    gen = torch.Generator().manual_seed(seed)
    ds_train, ds_val, ds_test = torch.utils.data.random_split(ds, [.7, .15, .15], generator=gen)
    return ds_train, ds_val, ds_test


def split_train_test_val2(ds: KelpDataset, seed: int = 42):
    """Split dataset by index, so sub datasets are subclass of KelpDataset. That allows to individually apply trafos."""
    # Get masks
    mask_train, mask_val, mask_test = get_train_val_test_masks(len(ds), seed)

    # Apply masks to dataset
    ds_train = KelpDataset(img_dir=ds.img_dir, mask_dir=ds.mask_dir, dir_mask=mask_train)
    ds_val = KelpDataset(img_dir=ds.img_dir, mask_dir=ds.mask_dir, dir_mask=mask_val)
    ds_test = KelpDataset(img_dir=ds.img_dir, mask_dir=ds.mask_dir, dir_mask=mask_test)

    return ds_train, ds_val, ds_test


def get_train_val_test_masks(n: int, random_seed: int = 42):
    """Split dataset by index, so sub datasets are subclass of KelpDataset. That allows to individually apply trafos."""
    # Randomly shuffle dataset indices
    rng = np.random.default_rng(seed=random_seed)
    inds = np.arange(n)
    rng.shuffle(inds)  # shuffle indices place

    # Train indices
    n_train = np.round(n * .7).astype(int)
    inds_train = inds[:n_train]
    inds_val_test = inds[n_train:]

    # Val and test indices
    n_val = (n - n_train) // 2
    inds_val = inds_val_test[:n_val]
    inds_test = inds_val_test[n_val:]

    # Make them all into masks
    inds = np.arange(n)  # new ordered index array
    mask_train = np.isin(inds, inds_train)
    mask_val = np.isin(inds, inds_val)
    mask_test = np.isin(inds, inds_test)

    return mask_train, mask_val, mask_test


if __name__ == "__main__":
    import tqdm
    ds_kwargs = {
        "img_nc_path": "data_ncf/train_imgs_fe.nc",
        "mask_nc_path": "data_ncf/train_masks.ncf",
        "n_rand_tiles": 25,
        "tile_size": 64,
        "random_seed": 42,
    }
    ds = KelpTiledDataset(**ds_kwargs)
    ds.load()

    # Compute total kelp count
    kelp_cnt = 0
    for _, mask in tqdm.tqdm(ds):
        has_kelp = mask.sum() > 0
        if has_kelp:
            kelp_cnt += 1
    print(kelp_cnt / len(ds))
