from glob import glob
import os

from ..config import cfg


def mk_res_dir():
    # Setup an experiment folder
    os.makedirs(cfg.results_dir, exist_ok=True)  # Create results folder if it doesn't exist

    # Determine the next experiment index
    experiment_index = max(
        (int(existing_dir.split("/")[-1].split("-")[0]) 
         for existing_dir in glob(f"{cfg.results_dir}/*") 
         if existing_dir.split("/")[-1].split("-")[0].isdigit()),
        default=-1
    ) + 1

    # Define experiment folder structure
    experiment_dir = os.path.join(cfg.results_dir, f"{experiment_index:03d}")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    samples_dir = os.path.join(experiment_dir, "samples")

    # Create necessary directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # Save a copy of the config file
    cfg.save(os.path.join(experiment_dir, "config.yaml"))

    return experiment_dir, checkpoint_dir, samples_dir
