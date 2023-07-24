from collections import defaultdict
from tqdm import trange
import wandb

# import dmc2gym
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
from rad.kitchen_train import compute_path_info
from rad.kitchen_spirl_pretrain import make_agent

render_imwidth, render_imheight = 256, 256


def evaluate(
    env,
    agent,
    num_episodes,
    encoder_type,
    data_augs,
    image_size,
    pre_transform_image_size,
    record_video,
):
    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        all_ep_rewards = []
        all_infos = []
        all_frames = {}
        for i in range(num_episodes):
            obs = env.reset()
            frames = []
            if record_video:
                frames.append(
                    env.render(
                        mode="rgb_array",
                        imwidth=render_imwidth,
                        imheight=render_imheight,
                    )
                )
            done = False
            episode_reward = 0
            ep_infos = []
            while not done:
                # center crop image
                if encoder_type == "pixel" and "crop" in data_augs:
                    obs = utils.center_crop_image(obs, image_size)
                if encoder_type == "pixel" and "translate" in data_augs:
                    # first crop the center with pre_transform_image_size
                    obs = utils.center_crop_image(obs, pre_transform_image_size)
                    # then translate cropped to center
                    obs = utils.center_translate(obs, image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, info = env.step(action)
                if record_video:
                    frames.append(
                        env.render(
                            mode="rgb_array",
                            imwidth=render_imwidth,
                            imheight=render_imheight,
                        )
                    )
                episode_reward += reward
                ep_infos.append(info)

            all_ep_rewards.append(episode_reward)
            all_infos.append(ep_infos)
            if record_video:
                all_frames["episode_%d" % i] = wandb.Video(
                    np.transpose(np.stack(frames), (0, 3, 1, 2)), fps=30, format="mp4"
                )
        eval_time = (time.time() - start_time) // num_episodes
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        std_ep_reward = np.std(all_ep_rewards)
        log_data = {
            "eval/average_return": mean_ep_reward,
            "eval/best_return": best_ep_reward,
            "eval/std_return": std_ep_reward,
            "eval/eval_time_per_ep": eval_time,
        }
        statistics = compute_path_info(all_infos)
        for k, v in statistics.items():
            log_data[f"eval/{k}"] = v
        for video_name, video in all_frames.items():
            log_data[f"eval/{video_name}"] = video
        return log_data

    log_data = run_eval_loop(sample_stochastically=False)
    return log_data


def experiment(variant):
    work_dir = rlkit_logger.get_snapshot_dir()
    seed = int(variant["seed"])
    utils.set_seed_everywhere(seed)
    os.makedirs(work_dir, exist_ok=True)
    agent_kwargs = variant["agent_kwargs"]
    data_augs = agent_kwargs["data_augs"]
    encoder_type = agent_kwargs["encoder_type"]
    env_suite = variant["env_suite"]
    env_name = variant["env_name"]
    env_kwargs = variant["env_kwargs"]
    pre_transform_image_size = variant["pre_transform_image_size"]
    image_size = variant["image_size"]
    frame_stack = variant["frame_stack"]
    batch_size = variant["batch_size"]
    spirl_latent_dim = variant["agent_kwargs"]["spirl_latent_dim"]
    # num_train_epochs = variant["num_train_epochs"]  # new arg
    run_group = variant["run_group"]
    replay_buffer_capacity = variant["replay_buffer_capacity"]
    num_train_steps = variant["num_train_steps"]
    num_eval_episodes = variant["num_eval_episodes"]
    eval_freq = variant["eval_freq"]
    init_steps = variant["init_steps"]
    pre_transform_image_size = (
        pre_transform_image_size if "crop" in data_augs else image_size
    )
    pre_transform_image_size = pre_transform_image_size

    if data_augs == "crop":
        pre_transform_image_size = 100
        image_size = image_size
    elif data_augs == "translate":
        pre_transform_image_size = 100
        image_size = 108

    if env_suite == "kitchen":
        env_kwargs["imwidth"] = pre_transform_image_size
        env_kwargs["imheight"] = pre_transform_image_size
    else:
        env_kwargs["image_kwargs"]["imwidth"] = pre_transform_image_size
        env_kwargs["image_kwargs"]["imheight"] = pre_transform_image_size
    expl_env = primitives_make_env.make_env(env_suite, env_name, env_kwargs)
    eval_env = primitives_make_env.make_env(env_suite, env_name, env_kwargs)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    env_name = env_name
    exp_name = (
        "SPiRL-online-"
        + env_name
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
    work_dir = os.path.join(work_dir, exp_name)
    utils.make_dir(work_dir)
    buffer_dir = utils.make_dir(os.path.join(work_dir, "buffer"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    discrete_action_dim = 0

    if encoder_type == "pixel":
        obs_shape = (3 * frame_stack, image_size, image_size)
        pre_aug_obs_shape = (
            3 * frame_stack,
            pre_transform_image_size,
            pre_transform_image_size,
        )
    else:
        obs_shape = expl_env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    if encoder_type == "pixel":
        obs_shape = (3 * frame_stack, image_size, image_size)
        pre_aug_obs_shape = (
            3 * frame_stack,
            pre_transform_image_size,
            pre_transform_image_size,
        )
    else:
        obs_shape = expl_env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_size=spirl_latent_dim + discrete_action_dim,
        capacity=replay_buffer_capacity,
        batch_size=batch_size,
        device=device,
        image_size=image_size,
        pre_image_size=pre_transform_image_size,
    )
    # load buffer if it already exists
    replay_buffer.load(buffer_dir)

    agent = make_agent(
        obs_shape=obs_shape,
        discrete_action_dim=discrete_action_dim,
        device=device,
        agent_kwargs=agent_kwargs,
    )

    agent = torch.compile(agent)

    wandb.init(
        project="p-amazon-intern", config=variant, name=exp_name, group=run_group
    )

    agent.train_spirl()  # not even necessary but just to be sure
    agent = agent.to(device)

    episode, episode_reward, hl_reward_sum, done = 0, 0, 0, True
    # start_time = time.time()
    # epoch_start_time = time.time()
    # train_expl_st = time.time()
    all_infos = []
    ep_infos = []
    num_train_calls = 0
    log_dict = defaultdict(list)
    hl_env_step_time = time.time()
    hl_steps = 0
    hl_policy_action = None
    for step in trange(num_train_steps):
        # evaluate agent periodically
        if step % eval_freq == 0:
            eval_start = time.time()
            eval_log_data = evaluate(
                eval_env,
                agent,
                num_eval_episodes,
                encoder_type,
                data_augs,
                image_size,
                pre_transform_image_size,
                record_video=True,
            )
            agent.save(work_dir, step)
            replay_buffer.save(buffer_dir)
            log_dict.update(eval_log_data)
            eval_end = time.time()
            log_dict["time/eval"].append(eval_end - eval_start)
            hl_env_step_time = time.time()  # reset hl env step because we're evalling
        if done:
            log_dict["train/episode_reward"].append(episode_reward)

            done = False
            episode_reward = 0
            hl_reward_sum = 0
            episode_step = 0
            episode += 1
            all_infos.append(ep_infos)

            log_dict["train/episode"].append(episode)
            statistics = compute_path_info(all_infos)
            for k, v in statistics.items():
                log_dict[f"train/{k}"] = v

            log_dict["trainer/num train calls"].append(num_train_calls)
            log_dict["trainer/num hl steps"].append(hl_steps)
            mean_log_dict = {}
            for k, v in log_dict.items():
                if not isinstance(v, wandb.Video):
                    mean_log_dict[k] = np.mean(v)
                else:
                    mean_log_dict[k] = v
            wandb.log(mean_log_dict, step=step * frame_stack)
            log_dict = defaultdict(list)
            obs = expl_env.reset()
            hl_obs = obs
            hl_policy_action = None

        # sample action for data collection
        # if step < init_steps:
        #    action = expl_env.action_space.sample()
        # else:

        with utils.eval_mode(agent):
            action = agent.sample_action(obs)
            if hl_policy_action is None:
                hl_policy_action = agent.current_latent.detach().cpu().numpy().flatten()
                agent._orig_mod.sampled_new_action = False  # don't add this current action to buffer because it's the first one

        training_log = {}

        next_obs, reward, done, info = expl_env.step(action)
        ep_infos.append(info)
        # allow infinit bootstrap
        done_bool = (
            0 if episode_step + 1 == expl_env._max_episode_steps else float(done)
        )
        episode_reward += reward
        hl_reward_sum += reward
        # orig_mod because we called torch.compile(agent) and for some reason the boolean isn't being updated
        if agent._orig_mod.sampled_new_action or done:
            hl_steps += 1
            log_dict["time/hl env step"].append(time.time() - hl_env_step_time)

            buffer_add_start = time.time()
            replay_buffer.add(
                hl_obs, hl_policy_action, hl_reward_sum, next_obs, done_bool
            )
            log_dict["time/add to buffer"].append(time.time() - buffer_add_start)

            hl_policy_action = agent.current_latent.detach().cpu().numpy().flatten()

            hl_obs = next_obs
            hl_reward_sum = 0
            # run training update
            rl_train_start = time.time()
            if step >= init_steps:
                num_updates = 1
                for _ in range(num_updates):
                    agent.update(replay_buffer, training_log, hl_steps)
                    num_train_calls += 1
            for k, v in training_log.items():
                log_dict[k].append(v)
            log_dict["time/train"].append(time.time() - rl_train_start)
            hl_env_step_time = time.time()

        obs = next_obs
        episode_step += 1
