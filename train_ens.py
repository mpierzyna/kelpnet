from typing import List
import concurrent.futures
import threading
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
import pathlib
import datetime
import numpy as np

ENS_ROOT = pathlib.Path("ens_seg") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_ROOT = ENS_ROOT / "logs"
LOG_ROOT.mkdir(parents=True, exist_ok=True)


def run(id_task: int, id_gpu: int) -> None:
    log_file_path = LOG_ROOT / f"task_{id_task}_log.txt"
    print(f"Starting task {id_task} on GPU {id_gpu}, logging to {log_file_path}.")

    with open(log_file_path, "w") as log_file:
        command = ["python", "torch_simple_unet.py", "--ens_root", str(ENS_ROOT), str(id_task), str(id_gpu)]
        p = subprocess.run(command, text=True, check=True, stdout=log_file, stderr=log_file)

    print(f"Task {id_task} on GPU {id_gpu} finished with return code {p.returncode}.")


def train_ensemble(n_members: int, n_gpus: int) -> None:
    gpu_in_use: np.ndarray = np.zeros(n_gpus, dtype=bool)
    lock: threading.Lock = threading.Lock()

    def acquire_gpu_and_run(id_task: int) -> None:
        # Find first available GPU
        with lock:
            gpu_id = np.where(~gpu_in_use)[0][0]
            gpu_in_use[gpu_id] = True

        # Run task
        run(id_task, gpu_id)

        # Release GPU
        with lock:
            gpu_in_use[gpu_id] = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = []
        for i in range(n_members):
            f = executor.submit(acquire_gpu_and_run, i)
            futures.append(f)
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    train_ensemble(n_members=25, n_gpus=3)
