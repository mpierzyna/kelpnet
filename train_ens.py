import concurrent.futures
import datetime
import pathlib
import subprocess
import threading
import time

import click
import numpy as np

mode_settings = {
    "seg": {
        "script": "torch_simple_unet.py",
        "ens_root": pathlib.Path("ens_seg"),
    },
    "clf": {
        "script": "torch_simple_clf.py",
        "ens_root": pathlib.Path("ens_clf"),
    },
}


def run(script: str, ens_dir: pathlib.Path, log_dir: pathlib.Path, id_task: int, id_gpu: int) -> None:
    # Random delay to make lightning race condition unlikely
    delay = np.random.uniform(0, 2)
    time.sleep(delay)

    log_file_path = log_dir / f"task_{id_task}_log.txt"
    print(f"Starting task {id_task} on GPU {id_gpu}, logging to {log_file_path}.")

    with open(log_file_path, "w") as log_file:
        command = ["python", script, "--ens_dir", str(ens_dir), str(id_task), str(id_gpu)]
        p = subprocess.run(command, text=True, check=True, stdout=log_file, stderr=log_file)

    print(f"Task {id_task} on GPU {id_gpu} finished with return code {p.returncode}.")


@click.command()
@click.option("--first-member", "offset", type=int, default=0)
@click.argument("mode", type=str)
@click.argument("n_members", type=int)
@click.argument("n_gpus", type=int)
def train_ensemble(offset: int, mode: str, n_members: int, n_gpus: int) -> None:
    if mode not in mode_settings:
        raise ValueError(f"Unknown mode {mode}.")

    # Setup result directories
    script = mode_settings[mode]["script"]
    ens_root = mode_settings[mode]["ens_root"]
    ens_dir = ens_root / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ens_dir.mkdir(parents=True, exist_ok=True)

    log_dir = ens_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_in_use: np.ndarray = np.zeros(n_gpus, dtype=bool)
    lock: threading.Lock = threading.Lock()

    def acquire_gpu_and_run(id_task: int) -> None:
        # Find first available GPU
        with lock:
            gpu_id = np.where(~gpu_in_use)[0][0]
            gpu_in_use[gpu_id] = True

        # Run task
        run(script, ens_dir, log_dir, id_task, gpu_id)

        # Release GPU
        with lock:
            gpu_in_use[gpu_id] = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = []
        for i in range(offset, n_members):
            f = executor.submit(acquire_gpu_and_run, i)
            futures.append(f)
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    train_ensemble()
