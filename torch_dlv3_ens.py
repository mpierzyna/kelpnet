import os

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor

import torch_deeplabv3 as dlv3

# Set the SLURM_JOB_ID environment variable to "debug" if not running on a cluster
ID = os.environ.get("SLURM_JOB_ID", "debug")
print("Using ID:", ID)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = dlv3.get_loaders(
        num_workers=8, batch_size=32, kf_weighing=True, random_seed=None  # No random seed!
    )

    # Train
    model = dlv3.LitDeepLabV3(n_ch=8, ens_prediction=True, lr=5e-4, lr_gamma=0.75, weight_decay=1e-1)
    trainer = L.Trainer(
        devices=3,
        max_epochs=20,  # 20 are enough
        log_every_n_steps=10,
        callbacks=[
            ModelCheckpoint(
                save_top_k=1,
                monitor="val_dice",
                mode="max",
                dirpath="dlv3_ens",
                filename=f"dlv3_{ID}" + "_{epoch:02d}_{val_dice:.2f}",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # New trainer on just one device
    trainer.test(ckpt_path="best", dataloaders=test_loader)
