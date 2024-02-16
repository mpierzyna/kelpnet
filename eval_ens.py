from typing import List
import pathlib

import lightning as L
import torch
import tqdm
import joblib

import torch_simple_unet as unet
from data import KelpTiledDataset


class EnsemblePredictor:
    def __init__(self, ckpt_files):
        self.used_ch = []
        for c in ckpt_files:
            _, _, used_ch, _, _, _ = c.stem.split("_")
            used_ch = used_ch.split("-")
            used_ch = [int(ch) for ch in used_ch]
            self.used_ch.append(used_ch)

        self.models = [
            unet.LitUNet.load_from_checkpoint(ckpt_file)
            for ckpt_file in tqdm.tqdm(ckpt_files, desc="Loading models")
        ]

    def predict(self, ds: KelpTiledDataset) -> List[torch.Tensor]:
        if ds.use_channels is not None:
            raise ValueError("Full dataset without channel selection is required.")

        trainer = L.Trainer(devices=1)

        y_hat = []
        for m, used_ch in zip(self.models, self.used_ch):
            ds.use_channels = used_ch
            loader = torch.utils.data.DataLoader(ds, batch_size=1024, num_workers=0, pin_memory=True)
            y_hat_i = trainer.predict(m, loader)
            y_hat_i = torch.cat(y_hat_i, dim=0)
            y_hat.append(y_hat_i)

        y_hat = torch.stack(y_hat)
        return y_hat


if __name__ == "__main__":
    # Load dataset to RAM
    _, _, ds_test = unet.get_dataset(use_channels=None, random_seed=1337)
    ds_test.load()

    # Prepare ensemble
    ckpt_files = sorted(pathlib.Path("ens_seg/20240214_101045").glob("*.ckpt"))
    ens = EnsemblePredictor(ckpt_files)

    # Make prediction
    y_hat = ens.predict(ds_test)
    joblib.dump(y_hat, "pred.joblib")
