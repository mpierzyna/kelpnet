from typing import List, Type, Dict, Tuple, Optional, Union
import pathlib

import click
import lightning as L
import torch
import tqdm
import joblib

import torch_simple_unet as unet
import torch_simple_clf as clf
import torch_deeplabv3 as dlv3
from data import KelpTiledDataset, KelpNCDataset
import shared


class EnsemblePredictor:
    def __init__(self, model_class: Type[L.LightningModule], ckpt_files: List[pathlib.Path],
                 devices: Optional[Union[int, List[int]]] = 1, **loader_kwargs) -> None:
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

        self.devices = devices
        self.loader_kwargs = {
            "batch_size": 1024,
            "num_workers": 0,
            "pin_memory": True,
            **loader_kwargs  # Override defaults
        }

    def test(self, ds: KelpNCDataset) -> Dict:
        # Only use single GPU for testing. Use last one in list if multiple are given.
        devices = self.devices[-1] if isinstance(self.devices, list) else self.devices
        trainer = L.Trainer(devices=devices)
        scores = []
        for m, used_ch in zip(self.models, self.used_ch):
            ds.use_channels = used_ch
            loader = torch.utils.data.DataLoader(ds, **self.loader_kwargs)
            scores_i = trainer.test(m, loader)
            scores.append(scores_i)

        return scores

    def predict(self, ds: KelpNCDataset) -> List[torch.Tensor]:
        trainer = L.Trainer(devices=self.devices)
        y_hat = []
        for m, used_ch in zip(self.models, self.used_ch):
            ds.use_channels = used_ch
            loader = torch.utils.data.DataLoader(ds, **self.loader_kwargs)
            y_hat_i = trainer.predict(m, loader)
            y_hat_i = torch.cat(y_hat_i, dim=0)
            y_hat.append(y_hat_i)

        y_hat = torch.stack(y_hat)
        return y_hat


@click.group()
@click.option("--test/--no-test", default=True)
@click.pass_context
def cli(ctx: click.Context, test):
    ctx.obj = {"test": test}


@cli.command(name="clf")
@click.argument("ens_dir", type=str)
@click.pass_obj
def make_clf_pred_cli(obj: Dict, ens_dir: str):
    run_test = obj.get("test", True)
    ens_dir = pathlib.Path(ens_dir)
    _, _, ds_test = shared.get_dataset(use_channels=None, split_seed=shared.GLOBAL_SEED, tile_seed=1337, mode="binary")

    joblib.dump(make_seg_pred(
        ens_dir=ens_dir,
        ds=ds_test,
        run_test=run_test
    ), ens_dir / "pred_clf.joblib")


def make_clf_pred(ens_dir: str, ds: KelpTiledDataset, run_test: bool) -> Tuple[Dict, torch.Tensor, List[int]]:
    # Load dataset to RAM
    ds.load()

    # Prepare ensemble
    ens_dir = pathlib.Path(ens_dir)
    ckpt_files = sorted(ens_dir.glob("*.ckpt"))
    ens = EnsemblePredictor(clf.LitBinaryClf, ckpt_files, batch_size=1024)

    # Make prediction
    if run_test:
        scores = ens.test(ds)
    else:
        scores = [None for _ in ckpt_files]
    y_hat = ens.predict(ds)

    return scores, y_hat, ens.used_ch


@cli.command(name="seg")
@click.argument("ens_dir", type=str)
@click.pass_obj
def make_seg_pred_cli(obj: Dict, ens_dir: str):
    run_test = obj.get("test", True)
    ens_dir = pathlib.Path(ens_dir)
    _, _, ds_test = shared.get_dataset(use_channels=None, split_seed=shared.GLOBAL_SEED, tile_seed=1337, mode="seg")

    joblib.dump(make_seg_pred(
        ens_dir=ens_dir,
        ds=ds_test,
        run_test=run_test
    ), ens_dir / "pred_seg.joblib")


def make_seg_pred(ens_dir: str, ds: KelpTiledDataset, run_test: bool) -> Tuple[Dict, torch.Tensor, List[int]]:
    # Load dataset to RAM
    ds.load()

    # Prepare ensemble
    ens_dir = pathlib.Path(ens_dir)
    ckpt_files = sorted(ens_dir.glob("*.ckpt"))
    ens = EnsemblePredictor(unet.LitUNet, ckpt_files, batch_size=1024)

    # Make prediction
    if run_test:
        scores = ens.test(ds)
    else:
        scores = [None for _ in ckpt_files]
    y_hat = ens.predict(ds)

    return scores, y_hat, ens.used_ch


@cli.command(name="dlv3")
@click.argument("ens_dir", type=str)
@click.pass_obj
def make_dlv3_pred_cli(obj: Dict, ens_dir: str):
    run_test = obj.get("test", True)
    ens_dir = pathlib.Path(ens_dir)
    _, _, ds_test = dlv3.get_dataset(use_channels=None, random_seed=shared.GLOBAL_SEED)

    joblib.dump(make_dlv3_pred(
        ens_dir=ens_dir,
        ds=ds_test,
        run_test=run_test
    ), ens_dir / "pred_dlv3.joblib")


def make_dlv3_pred(ens_dir: str, ds: KelpTiledDataset, run_test: bool) -> Tuple[Dict, torch.Tensor, List[int]]:
    # Prepare ensemble
    ens_dir = pathlib.Path(ens_dir)
    ckpt_files = sorted(ens_dir.glob("*.ckpt"))
    ens = EnsemblePredictor(dlv3.LitDeepLabV3, ckpt_files, batch_size=32, num_workers=8, devices=[1, 2])

    # Make prediction
    if run_test:
        scores = ens.test(ds)
    else:
        scores = [None for _ in ckpt_files]
    y_hat = ens.predict(ds)

    return scores, y_hat, ens.used_ch


if __name__ == "__main__":
    cli()
