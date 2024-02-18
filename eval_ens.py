from typing import List, Type, Dict
import pathlib

import click
import lightning as L
import torch
import tqdm
import joblib

import torch_simple_unet as unet
import torch_simple_clf as clf
from data import KelpTiledDataset, KelpNCDataset
import shared


class EnsemblePredictor:
    def __init__(self, model_class: Type[L.LightningModule], ckpt_files, batch_size: int):
        self.used_ch = []
        for c in ckpt_files:
            _, _, used_ch, _, _, _ = c.stem.split("_")
            used_ch = used_ch.split("-")
            used_ch = [int(ch) for ch in used_ch]
            self.used_ch.append(used_ch)

        self.models = [
            model_class.load_from_checkpoint(ckpt_file)
            for ckpt_file in tqdm.tqdm(ckpt_files, desc="Loading models")
        ]

        self.batch_size = batch_size

    def test(self, ds: KelpNCDataset) -> Dict:
        trainer = L.Trainer(devices=1)
        scores = []
        for m, used_ch in zip(self.models, self.used_ch):
            ds.use_channels = used_ch
            loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, num_workers=0, pin_memory=True)
            scores_i = trainer.test(m, loader)
            scores.append(scores_i)

        return scores

    def predict(self, ds: KelpNCDataset) -> List[torch.Tensor]:
        trainer = L.Trainer(devices=1)
        y_hat = []
        for m, used_ch in zip(self.models, self.used_ch):
            ds.use_channels = used_ch
            loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, num_workers=0, pin_memory=True)
            y_hat_i = trainer.predict(m, loader)
            y_hat_i = torch.cat(y_hat_i, dim=0)
            y_hat.append(y_hat_i)

        y_hat = torch.stack(y_hat)
        return y_hat


@click.group()
def cli():
    ...


@cli.command()
def make_clf_pred():
    # Load dataset to RAM
    _, _, ds_test = shared.get_dataset(use_channels=None, split_seed=shared.GLOBAL_SEED, tile_seed=1337, mode="binary")
    # _, _, ds_test = shared.get_untiled_dataset(use_channels=None, split_seed=shared.GLOBAL_SEED, mode="binary")
    ds_test.load()

    # Prepare ensemble
    ckpt_files = sorted(pathlib.Path("ens_clf/20240216_041023").glob("*.ckpt"))
    ens = EnsemblePredictor(clf.LitBinaryClf, ckpt_files, batch_size=1024)

    # Make prediction
    scores = ens.test(ds_test)
    y_hat = ens.predict(ds_test)
    joblib.dump([scores, y_hat], "pred_clf.joblib")


@cli.command()
def make_seg_pred():
    # Load dataset to RAM
    _, _, ds_test = shared.get_dataset(use_channels=None, split_seed=shared.GLOBAL_SEED, tile_seed=1337, mode="seg")
    # _, _, ds_test = shared.get_untiled_dataset(use_channels=None, split_seed=shared.GLOBAL_SEED, mode="binary")
    ds_test.load()

    # Prepare ensemble
    ckpt_files = sorted(pathlib.Path("ens_seg/20240216_095221").glob("*.ckpt"))
    ens = EnsemblePredictor(unet.LitUNet, ckpt_files, batch_size=1024)

    # Make prediction
    scores = ens.test(ds_test)
    y_hat = ens.predict(ds_test)
    joblib.dump([scores, y_hat], "pred_seg.joblib")


if __name__ == "__main__":
    # make_clf_pred()
    # make_seg_pred()
    cli()
