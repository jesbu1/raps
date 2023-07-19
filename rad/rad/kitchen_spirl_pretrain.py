import argparse
from collections import defaultdict
from tqdm import trange
import wandb

# import dmc2gym
import copy
import json
import os
import time
from collections import OrderedDict

import gym
import numpy as np
import d4rl
import rlkit.pythonplusplus as ppp
import torch
from rlkit.core import logger as rlkit_logger
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.envs.primitives_make_env import (
    make_base_kitchen_env,
    make_base_metaworld_env,
)
from rlkit.envs.primitives_wrappers import (
    ActionRepeat,
    ImageEnvMetaworld,
    ImageUnFlattenWrapper,
    NormalizeActions,
    TimeLimit,
)
from torchvision import transforms
import rlkit.envs.primitives_make_env as primitives_make_env
import rad.utils as utils
from rad.curl_spirl import SPiRLRadSacAgent
from rad.logger import Logger
from rad.video import VideoRecorder


def make_agent(
    obs_shape,
    continuous_action_dim,
    discrete_action_dim,
    agent_kwargs,
    device,
):
    return SPiRLRadSacAgent(
        obs_shape=obs_shape,
        continuous_action_dim=continuous_action_dim,
        discrete_action_dim=discrete_action_dim,
        # env_action_dim=agent_kwargs["env_action_dim"],
        device=device,
        **agent_kwargs,
    )


def experiment(variant):
    work_dir = rlkit_logger.get_snapshot_dir()
    seed = int(variant["seed"])
    utils.set_seed_everywhere(seed)
    os.makedirs(work_dir, exist_ok=True)
    agent_kwargs = variant["agent_kwargs"]
    data_augs = agent_kwargs["data_augs"]
    encoder_type = agent_kwargs["encoder_type"]
    # discrete_continuous_dist = agent_kwargs["discrete_continuous_dist"]

    # env_suite = variant["env_suite"]
    env_name = variant["env_name"]
    # env_kwargs = variant["env_kwargs"]
    # pre_transform_image_size = variant["pre_transform_image_size"]
    image_size = "state"  # variant["image_size"]
    frame_stack = variant["frame_stack"]
    batch_size = variant["batch_size"]
    num_train_epochs = variant["num_train_epochs"]  # new arg
    spirl_skill_len = variant["agent_kwargs"]["spirl_action_horizon"]
    run_group = variant["run_group"]
    # replay_buffer_capacity = variant["replay_buffer_capacity"]
    # num_train_steps = variant["num_train_steps"]
    # num_eval_episodes = variant["num_eval_episodes"]
    # eval_freq = variant["eval_freq"]
    # log_interval = variant["log_interval"]
    # use_raw_actions = variant["use_raw_actions"]
    # pre_transform_image_size = (
    #    pre_transform_image_size if "crop" in data_augs else image_size
    # )
    # pre_transform_image_size = pre_transform_image_size

    # if data_augs == "crop":
    #    pre_transform_image_size = 100
    #    image_size = image_size
    # elif data_augs == "translate":
    #    pre_transform_image_size = 100
    #    image_size = 108

    # if env_suite == "kitchen":
    #    env_kwargs["imwidth"] = pre_transform_image_size
    #    env_kwargs["imheight"] = pre_transform_image_size
    # else:
    #    env_kwargs["image_kwargs"]["imwidth"] = pre_transform_image_size
    #    env_kwargs["image_kwargs"]["imheight"] = pre_transform_image_size
    env = gym.make(env_name)
    d4rl_dataset = env.get_dataset()

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    env_name = env_name
    exp_name = (
        env_name
        + "-"
        + ts
        + "-im"
        + str(image_size)
        + "-b"
        + str(batch_size)
        + "-s"
        + str(seed)
        + "-"
        + encoder_type
    )
    work_dir = work_dir + "/" + exp_name

    utils.make_dir(work_dir)
    # video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    # model_dir = utils.make_dir(os.path.join(work_dir, "model"))
    # buffer_dir = utils.make_dir(os.path.join(work_dir, "buffer"))

    # video = VideoRecorder(video_dir if args.save_video else None)

    # with open(os.path.join(work_dir, "args.json"), "w") as f:
    #    json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    continuous_action_dim = env.action_space.low.size
    discrete_action_dim = 0

    if encoder_type == "pixel":
        obs_shape = (3 * frame_stack, image_size, image_size)
        pre_aug_obs_shape = (
            3 * frame_stack,
            pre_transform_image_size,
            pre_transform_image_size,
        )
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    d4rl_dataset = env.get_dataset()

    spirl_dataset = utils.D4RLSequenceSplitDataset(
        batch_size=batch_size,
        device=device,
        d4rl_dataset=d4rl_dataset,
        skill_len=spirl_skill_len,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        continuous_action_dim=continuous_action_dim,
        discrete_action_dim=discrete_action_dim,
        device=device,
        agent_kwargs=agent_kwargs,
    )

    agent = torch.compile(agent)

    wandb.init(
        project="p-amazon-intern", config=variant, name=exp_name, group=run_group
    )

    # L = Logger(work_dir, use_tb=args.save_tb)

    agent.train_spirl()  # not even necessary but just to be sure
    agent = agent.to(device)
    for epoch in trange(num_train_epochs):
        epoch_log_dict = defaultdict(list)
        epoch_start_time = time.time()
        for step in range(int(len(spirl_dataset) / batch_size)):
            log_dict = agent.spirl_update(spirl_dataset, step)
            for k, v in log_dict.items():
                epoch_log_dict[k].append(v)
        epoch_end_time = time.time()
        epoch_log_dict = {k: np.mean(v) for k, v in epoch_log_dict.items()}
        epoch_log_dict["epoch"] = epoch
        epoch_log_dict["time/epoch (s)"] = epoch_end_time - epoch_start_time

    # save the checkpoint
    agent.save_spirl(work_dir, num_train_epochs)
