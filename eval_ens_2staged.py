"""
Function and variable suffixes:
- _aa: returning/containing original sized images (350, 350)
- _bb: returning/containing tiles of images (64, 64)
"""
import pathlib

import joblib
import numpy as np
import torch

import data
import shared
from eval_ens import EnsemblePredictor
import torchmetrics


def get_sea_mask_aa(ds: data.KelpTiledDataset) -> np.ndarray:
    """Get sea mask from dataset."""
    print("Getting sea mask...")
    is_land = ds.imgs.isel(ch=data.Channel.IS_LAND).astype(bool)
    is_sea = ~is_land
    return is_sea.to_numpy().astype(np.uint8)


def agg_bb_ens_pred(y_hat_bb: torch.Tensor, ts: data.RegularTileSampler) -> np.ndarray:
    print("pred:", y_hat_bb.shape, y_hat_bb.dtype)

    # Majority voting
    y_hat_bb = (y_hat_bb > 0.5).float()  # per member (m, n_bb, 1) or (m, n_bb, b, b)
    y_hat_bb = (y_hat_bb.mean(axis=0) > 0.5).float()  # aggregate (n_bb, 1) or (n_bb, b, b)
    print("pred agg:", y_hat_bb.shape, y_hat_bb.dtype)

    # Reassamble tiles to original size
    # n_aa: number of original sized images
    n_aa, remainder = np.divmod(len(y_hat_bb), ts.n_tiles)
    if remainder != 0:
        raise ValueError(f"Number of tiles {ts.n_tiles} does not divide original size {len(y_hat_bb)}.")

    # Empty result arrays
    y_hat_aa = np.zeros((n_aa, ts.orig_size_, ts.orig_size_), dtype=np.float32)
    count_aa = np.zeros((n_aa, ts.orig_size_, ts.orig_size_), dtype=np.float32)

    # i_aa: index of aa mask
    # i_bb_offset: offset for group of tiles that form aa mask [0, n_tiles]
    # i_bb: tile relative to offset to be processed
    # (i, j): tile indices in aa mask to be filled
    for i_aa, i_bb_offset in enumerate(range(0, len(y_hat_bb), ts.n_tiles)):
        for i_bb, (i, j) in enumerate(ts.inds_):
            tile = y_hat_bb[i_bb_offset + i_bb].numpy()  # float (1, ) or (b, b)
            y_hat_aa[i_aa, i:i + ts.tile_size, j:j + ts.tile_size] += tile
            count_aa[i_aa, i:i + ts.tile_size, j:j + ts.tile_size] += 1

    assert np.all(count_aa > 0)
    print("pred agg aa:", y_hat_aa.shape, y_hat_aa.dtype)

    # Normalize overlapping predictions
    y_hat_aa /= count_aa

    return y_hat_aa


def get_kelp_clf_mask_aa(clf_ens_dir: pathlib.Path, ts: data.RegularTileSampler) -> np.ndarray:
    print("Processing clf prediction...")
    if not (clf_ens_dir / "pred_clf.joblib").exists():
        raise ValueError(f"Predictions not found in {clf_ens_dir}.")

    # Load precomputed predictions
    y_hat_bb: torch.Tensor
    _, y_hat_bb, _ = joblib.load(clf_ens_dir / "pred_clf.joblib")

    # Aggregate ensembe and tiles
    y_hat_aa = agg_bb_ens_pred(y_hat_bb, ts)
    y_hat_aa = np.ceil(y_hat_aa)  # If in doubt, consider as kelp
    print("pred kelp fraction", y_hat_aa.sum(-1).sum(-1) / ts.orig_size_ ** 2)

    return y_hat_aa


def get_kelp_seg_mask_aa(seg_ens_dir: pathlib.Path, ts: data.RandomTileSampler):
    print("Processing seg prediction...")
    if not (seg_ens_dir / "pred_seg.joblib").exists():
        raise ValueError(f"Predictions not found in {seg_ens_dir}.")

    # Load precomputed predictions
    y_hat_bb: torch.Tensor
    _, y_hat_bb, _ = joblib.load(seg_ens_dir / "pred_seg.joblib")

    # Aggregate ensembe and tiles
    y_hat_aa = agg_bb_ens_pred(y_hat_bb, ts)
    y_hat_aa = (y_hat_aa > 0.5).astype(np.float32)
    print("pred kelp fraction", y_hat_aa.sum(-1).sum(-1) / ts.orig_size_ ** 2)

    return agg_bb_ens_pred(y_hat_bb, ts)


def get_2staged_kelp_mask_aa(clf_ens_dir: pathlib.Path, seg_ens_dir: pathlib.Path, ds: data.KelpTiledDataset) -> np.ndarray:
    # Aggregate tile-based predictions of classifier (clf) and segmentation model (seg)
    y_hat_clf_aa = get_kelp_clf_mask_aa(clf_ens_dir, ts=ds.tile_sampler)
    y_hat_seg_aa = get_kelp_seg_mask_aa(seg_ens_dir, ts=ds.tile_sampler)

    # Get sea mask
    ds.load()
    is_sea = get_sea_mask_aa(ds)

    # Postprocessing
    y_hat_seg_aa = y_hat_seg_aa * y_hat_clf_aa  # Remove non-kelp predictions
    y_hat_seg_aa = y_hat_seg_aa * is_sea  # Remove non-sea predictions
    return y_hat_seg_aa


def main():
    # ds = shared.get_submission_dataset()

    clf_ens_dir = pathlib.Path("ens_clf/20240216_041023")
    seg_ens_dir = pathlib.Path("ens_seg/20240219_163535")
    _, _, ds = shared.get_dataset(use_channels=None, split_seed=shared.GLOBAL_SEED, tile_seed=1337, mode="seg")

    y_hat_aa = get_2staged_kelp_mask_aa(clf_ens_dir, seg_ens_dir, ds)
    y_true_aa = ds.masks.to_numpy()
    score = torchmetrics.functional.dice(
        preds=torch.from_numpy(y_hat_aa),
        target=torch.from_numpy(y_true_aa),
    )
    print("Dice score:", score)

    print("Saving prediction...")
    joblib.dump(y_hat_aa, "pred_2staged.joblib")


if __name__ == "__main__":
    main()
