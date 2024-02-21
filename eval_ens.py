import pathlib
from typing import Dict, List, Tuple, Type

import click
import joblib
import lightning as L
import torch

import shared
import torch_deeplabv3 as dlv3
import torch_simple_clf as clf
import torch_simple_unet as unet
from data import KelpNCDataset, KelpTiledDataset
from shared import PredMode


class EnsemblePredictor:
    def __init__(self, model_class: Type[L.LightningModule], ckpt_files: List[pathlib.Path], device: int, **loader_kwargs) -> None:
        self.used_ch = []
        for c in ckpt_files:
            _, _, used_ch, _, _, _ = c.stem.split("_")
            used_ch = used_ch.split("-")
            used_ch = [int(ch) for ch in used_ch]
            self.used_ch.append(used_ch)

        self.model_class = model_class
        self.ckpt_files = ckpt_files
        self.device = device
        self.loader_kwargs = {
            "batch_size": 1024,
            "num_workers": 0,
            "pin_memory": True,
            **loader_kwargs  # Override defaults
        }

    def get_val_scores(self) -> List[float]:
        """Get validation score from checkpoint."""
        scores = []
        for ckpt in self.ckpt_files:
            ckpt = torch.load(ckpt, map_location="cpu")

            # Find key for ModelCheckpoint callback
            cb = ckpt["callbacks"]
            cb_key = None
            for k in cb.keys():
                if "ModelCheckpoint" in k:
                    cb_key = k
                break

            if cb_key is None:
                scores.append(None)
            else:
                scores.append(float(cb[cb_key]["best_model_score"]))

        return scores

    def test(self, ds: KelpNCDataset) -> Dict:
        raise NotImplementedError()

    def predict(self, ds: KelpNCDataset) -> List[torch.Tensor]:
        trainer = L.Trainer(devices=[self.device])
        y_hat = []
        for ckpt_path, used_ch in zip(self.ckpt_files, self.used_ch):
            print(f"Processing {ckpt_path.stem}...")

            # Prepare dataset with channels used for training
            ds.use_channels = used_ch
            loader = torch.utils.data.DataLoader(ds, **self.loader_kwargs)

            # Load model from checkpoint and make prediction
            m = self.model_class.load_from_checkpoint(ckpt_path)
            y_hat_i = trainer.predict(m, loader)
            y_hat_i = torch.cat(y_hat_i, dim=0)
            y_hat.append(y_hat_i)

        y_hat = torch.stack(y_hat)
        return y_hat


@click.group()
@click.option("--test/--no-test", default=False, is_flag=True)
@click.option("--device", type=int, default=0)
@click.argument("mode", type=PredMode, required=True)
@click.pass_context
def cli(ctx: click.Context, test: bool, device: int, mode: PredMode):
    ctx.obj = {
        "test": test,
        "mode": mode,
        "device": device,
    }


@cli.command(name="clf")
@click.argument("ens_dir", type=str)
@click.pass_obj
def make_clf_pred_cli(obj: Dict, ens_dir: str):
    mode = obj.get("mode")
    ens_dir = pathlib.Path(ens_dir)
    if mode is PredMode.TEST:
        _, _, ds = shared.get_dataset(use_channels=None, split_seed=shared.GLOBAL_SEED, tile_seed=1337, mode="binary")
    elif mode is PredMode.SUBMISSION:
        ds = shared.get_submission_dataset()
    else:
        raise ValueError(f"Unknown mode {mode}.")

    joblib.dump(
        make_clf_pred(ens_dir=ens_dir, ds=ds, run_test=obj["test"], device=obj["device"]),
        ens_dir / f"pred_clf_{mode}.joblib"
    )


def make_clf_pred(ens_dir: str, ds: KelpTiledDataset, run_test: bool, device: int = 0) -> Tuple[Dict, torch.Tensor, List[int]]:
    # Load dataset to RAM
    ds.load()

    # Prepare ensemble
    ens_dir = pathlib.Path(ens_dir)
    ckpt_files = sorted(ens_dir.glob("*.ckpt"))
    ens = EnsemblePredictor(clf.LitBinaryClf, ckpt_files, batch_size=1024, device=device)

    # Make prediction
    if run_test:
        scores = ens.test(ds)
    else:
        scores = ens.get_val_scores()
    y_hat = ens.predict(ds)

    return scores, y_hat, ens.used_ch


@cli.command(name="seg")
@click.argument("ens_dir", type=str)
@click.pass_obj
def make_seg_pred_cli(obj: Dict, ens_dir: str):
    mode = obj.get("mode")
    ens_dir = pathlib.Path(ens_dir)
    if mode is PredMode.TEST:
        _, _, ds = shared.get_dataset(use_channels=None, split_seed=shared.GLOBAL_SEED, tile_seed=1337, mode="binary")
    elif mode is PredMode.SUBMISSION:
        ds = shared.get_submission_dataset()
    else:
        raise ValueError(f"Unknown mode {mode}.")

    joblib.dump(
        make_seg_pred(ens_dir=ens_dir, ds=ds, run_test=obj["test"], device=obj["device"]),
        ens_dir / f"pred_seg_{mode}.joblib"
    )


def make_seg_pred(ens_dir: str, ds: KelpTiledDataset, run_test: bool, device: int = 0) -> Tuple[Dict, torch.Tensor, List[int]]:
    # Load dataset to RAM
    ds.load()

    # Prepare ensemble
    ens_dir = pathlib.Path(ens_dir)
    ckpt_files = sorted(ens_dir.glob("*.ckpt"))
    ens = EnsemblePredictor(unet.LitUNet, ckpt_files, batch_size=1024, device=device)

    # Make prediction
    if run_test:
        scores = ens.test(ds)
    else:
        scores = ens.get_val_scores()
    y_hat = ens.predict(ds)

    return scores, y_hat, ens.used_ch


@cli.command(name="dlv3")
@click.argument("ens_dir", type=str)
@click.pass_obj
def make_dlv3_pred_cli(obj: Dict, ens_dir: str):
    mode = obj.get("mode")
    ens_dir = pathlib.Path(ens_dir)
    if mode is PredMode.TEST:
        _, _, ds = dlv3.get_dataset(use_channels=None, random_seed=shared.GLOBAL_SEED)
    elif mode is PredMode.SUBMISSION:
        ds = dlv3.get_submission_dataset()
    else:
        raise ValueError(f"Unknown mode {mode}.")

    joblib.dump(
        make_dlv3_pred(ens_dir=ens_dir, ds=ds, run_test=obj["test"], device=obj["device"]),
        ens_dir / f"pred_dlv3_{mode}.joblib"
    )


def make_dlv3_pred(ens_dir: str, ds: KelpTiledDataset, run_test: bool, device: int = 0) -> Tuple[Dict, torch.Tensor, List[int]]:
    # Prepare ensemble
    ens_dir = pathlib.Path(ens_dir)
    ckpt_files = sorted(ens_dir.glob("*.ckpt"))
    ens = EnsemblePredictor(dlv3.LitDeepLabV3, ckpt_files, batch_size=32, num_workers=8, device=device)

    # Make prediction
    if run_test:
        scores = ens.test(ds)
    else:
        scores = ens.get_val_scores()
    y_hat = ens.predict(ds)

    return scores, y_hat, ens.used_ch


if __name__ == "__main__":
    cli()
