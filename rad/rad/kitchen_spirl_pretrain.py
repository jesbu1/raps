import argparse

# import dmc2gym
import copy
import json
import os
import time
from collections import OrderedDict

import gym
import numpy as np
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


def parse_args():
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument("--agent", default="rad_sac", type=str)
    parser.add_argument("--hidden_dim", default=1024, type=int)
    # critic
    parser.add_argument("--critic_beta", default=0.9, type=float)
    parser.add_argument("--critic_tau", default=0.01, type=float)  # try 0.05 or 0.1
    parser.add_argument(
        "--critic_target_update_freq", default=2, type=int
    )  # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument("--actor_beta", default=0.9, type=float)
    parser.add_argument("--actor_log_std_min", default=-10, type=float)
    parser.add_argument("--actor_log_std_max", default=2, type=float)
    parser.add_argument("--actor_update_freq", default=2, type=int)
    parser.add_argument("--discrete_continuous_dist", default=0, type=int)
    # encoder
    parser.add_argument("--encoder_feature_dim", default=50, type=int)
    parser.add_argument("--encoder_tau", default=0.05, type=float)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--num_filters", default=32, type=int)
    parser.add_argument("--latent_dim", default=128, type=int)
    # sac
    parser.add_argument("--init_temperature", default=0.1, type=float)
    parser.add_argument("--alpha_lr", default=1e-4, type=float)
    parser.add_argument("--alpha_beta", default=0.5, type=float)
    # misc
    parser.add_argument("--save_tb", default=False, action="store_true")
    parser.add_argument("--save_buffer", default=False, action="store_true")
    parser.add_argument("--save_video", default=False, action="store_true")
    parser.add_argument("--save_model", default=False, action="store_true")
    parser.add_argument("--detach_encoder", default=False, action="store_true")
    args = parser.parse_args()
    return args


def make_agent(
    obs_shape,
    continuous_action_dim,
    discrete_action_dim,
    args,
    agent_kwargs,
    device,
):
    if args.agent == "rad_spirl":
        return SPiRLRadSacAgent(
            obs_shape=obs_shape,
            continuous_action_dim=continuous_action_dim,
            discrete_action_dim=discrete_action_dim,
            env_action_dim=args.env_action_dim,
            device=device,
            hidden_dim=args.hidden_dim,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            latent_dim=args.latent_dim,
            **agent_kwargs,
        )
    else:
        assert "agent is not supported: %s" % args.agent


def experiment(variant):
    work_dir = rlkit_logger.get_snapshot_dir()
    args = parse_args()
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
    pre_transform_image_size = variant["pre_transform_image_size"]
    image_size = variant["image_size"]
    frame_stack = variant["frame_stack"]
    batch_size = variant["batch_size"]
    num_train_epochs = variant["num_train_epochs"]  # new arg
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

    if data_augs == "crop":
        pre_transform_image_size = 100
        image_size = image_size
    elif data_augs == "translate":
        pre_transform_image_size = 100
        image_size = 108

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
    video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    model_dir = utils.make_dir(os.path.join(work_dir, "model"))
    buffer_dir = utils.make_dir(os.path.join(work_dir, "buffer"))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(work_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

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
        skill_len=args.spirl_skill_len,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        continuous_action_dim=continuous_action_dim,
        discrete_action_dim=discrete_action_dim,
        args=args,
        device=device,
        agent_kwargs=agent_kwargs,
    )

    L = Logger(work_dir, use_tb=args.save_tb)

    start_time = time.time()
    agent.train_spirl()  # not even necessary but just to be sure
    for epoch in range(num_train_epochs):
        epoch_start_time = time.time()
        for step in range(int(len(spirl_dataset) / batch_size)):
            agent.spirl_update(spirl_dataset, L, step)
        epoch_end_time = time.time()
        L.log("train/epoch_time", epoch_end_time - epoch_start_time, epoch)
    L.log("train/total_time", time.time() - start_time, epoch)
