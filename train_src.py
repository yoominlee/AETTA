import os
import subprocess
import time
from itertools import cycle

# Constants
LOG_PREFIX = "reproduce_src"
DATASETS = ["imagenet"]  # Options: "cifar10", "cifar100"
METHODS = ["Src"]
GPUS = list(range(1))  # Assuming 8 GPUs available #원래 8
NUM_GPUS = len(GPUS)
DEFAULT_NUM_JOBS = 2 # 원래 8
RAW_LOGS_DIR = "raw_logs"

# Create a directory to save logs if it doesn't exist
os.makedirs(RAW_LOGS_DIR, exist_ok=True)

def wait_n(num_max_jobs=DEFAULT_NUM_JOBS):
    print('-- START -- wait_n')
    """ Wait for subprocesses to finish until only `num_max_jobs` are left. """
    while True:
        active_processes = [p for p in all_processes if p.poll() is None]
        if len(active_processes) < num_max_jobs:
            break
        time.sleep(1)
    print('-- END -- wait_n')

def train_source_model():
    print('-- START -- train_source_model')
    i = 0
    update_every_x = "64"
    memory_size = "64"
    gpu_cycle = cycle(GPUS)

    for dataset in DATASETS:
        for method in METHODS:
            validation = "--dummy"

            if dataset in ["cifar10", "cifar10outdist"]:
                epoch = 200
                model = "resnet18"
                tgt = "test"
            elif dataset == "cifar100":
                epoch = 200
                model = "resnet18"
                tgt = "test"
            elif dataset == "imagenet":
                epoch = 30
                model = "resnet18_pretrained"
                tgt = "test"
            elif dataset == "es":
                epoch = 20
                model = "resnet18_es_pretrained"
                tgt = "test"

            for seed in range(3):
                if "Src" in method:
                    for tgt in [tgt]:
                        cmd = [
                            "python", "main.py",
                            "--gpu_idx", str(next(gpu_cycle)),
                            "--dataset", dataset,
                            "--method", method,
                            "--tgt", tgt,
                            "--model", model,
                            "--epoch", str(epoch),
                            "--update_every_x", update_every_x,
                            "--memory_size", memory_size,
                            "--seed", str(seed),
                            "--log_prefix", f"{LOG_PREFIX}_{seed}",
                            validation
                        ]
                        log_file_path = os.path.join(RAW_LOGS_DIR, f"{dataset}_{LOG_PREFIX}_{seed}_job{i}.txt")
                        with open(log_file_path, "w") as log_file:
                            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                            for line in process.stdout:
                                log_file.write(line.decode())
                                print(line.decode(), end='')
                        all_processes.append(process)
                        i += 1
                        wait_n()


    print('-- END --train_source_model')

if __name__ == "__main__":
    all_processes = []
    train_source_model()
    for process in all_processes:
        process.wait()  # Ensure all processes are finished
