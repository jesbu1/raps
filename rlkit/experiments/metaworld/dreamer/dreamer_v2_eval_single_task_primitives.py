import argparse
import os
import pickle
import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant,
)
from rlkit.torch.model_based.dreamer.experiments.kitchen_dreamer import experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument('--load_dir', type=str, default=None, required=True)
    parser.add_argument("--env", type=str, default="")
    args = parser.parse_args()

    # load the variant
    saved_experiment_path = os.path.join(args.load_dir, "experiment.pkl")
    with open(saved_experiment_path, "rb") as f:
        saved_experiment = pickle.load(f)['run_experiment_here_kwargs']
    # modify the variant a little
    saved_experiment["variant"]["checkpoint_path"] = args.checkpoint_path
    saved_experiment["variant"]["log_dir"] = args.load_dir
    saved_experiment["variant"]["num_expl_envs"] = 1
    #search_space = {"env_name": [args.env]}

    variant = preprocess_variant(saved_experiment["variant"], debug=False)
    for _ in range(args.num_seeds):
        seed = random.randint(0, 100000)
        variant["seed"] = seed
        #variant["exp_id"] = 0
        eval_experiment(
            experiment,
            exp_prefix=args.exp_prefix,
            mode=args.mode,
            variant=variant,
            use_gpu=True,
            snapshot_mode="last",
            python_cmd=subprocess.check_output("which python", shell=True).decode(
                "utf-8"
            )[:-1],
            seed=seed,
            #exp_id=exp_id,
        )
