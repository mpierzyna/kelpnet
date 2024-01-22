import pathlib
import numpy as np
import rasterio
import pandas as pd

from torch.utils.data import Dataset


def load_img(fpath_img, fpath_mask):
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
    
    # Load mask
    mask = rasterio.open(fpath_mask).read()
    mask = mask[0].astype(np.uint8)
    
    return img, mask


class KelpDataset(Dataset):
    def __init__(self, img_dir, mask_dir, dir_mask = None):
        # List files
        img_list = sorted(pathlib.Path(img_dir).glob("*.tif"))
        mask_list = sorted(pathlib.Path(mask_dir).glob("*.tif"))
        
        # Sanity check that images and labels are associated correctly
        for img_path, mask_path in zip(img_list, mask_list):
            img_tile_id, _ = img_path.stem.split("_")
            mask_tile_id, _ = mask_path.stem.split("_")
            assert img_tile_id == mask_tile_id
        
        # Convert to np array of strings
        def path_to_str_array(path_list):
            return np.array([str(p) for p in path_list])

        self.img_list = path_to_str_array(img_list)
        self.mask_list = path_to_str_array(mask_list)
        if dir_mask is not None:
            self.img_list = self.img_list[dir_mask]
            self.mask_list = self.mask_list[dir_mask]
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
