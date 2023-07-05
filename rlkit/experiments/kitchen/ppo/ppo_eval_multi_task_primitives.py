import argparse
import pickle
import os
import random
import subprocess

import numpy as np
import torch

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment


def experiment(variant):
    from a2c_ppo_acktr.main import eval_experiment

    eval_experiment(variant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="ppo_eval")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument('--load_dir', type=str, default=None, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None, required=True)
    args = parser.parse_args()
    exp_prefix = args.exp_prefix
    saved_experiment_path = os.path.join(args.load_dir, "experiment.pkl")
    with open(saved_experiment_path, "rb") as f:
        saved_experiment = pickle.load(f)['run_experiment_here_kwargs']
    # modify the variant a little
    saved_experiment["variant"]["checkpoint_path"] = args.checkpoint_path
    saved_experiment["variant"]["log_dir"] = args.load_dir
    saved_experiment["variant"]["num_processes"] = 1
    

    for _ in range(args.num_seeds):
        seed = random.randint(0, 100000)
        run_experiment(
            experiment,
            exp_prefix=args.exp_prefix,
            mode=args.mode,
            variant=saved_experiment["variant"],
            use_gpu=True,
            snapshot_mode="last",
            python_cmd=subprocess.check_output("which python", shell=True).decode(
                "utf-8"
            )[:-1],
            seed=seed,
        )
