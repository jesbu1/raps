import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time
from skimage.util.shape import view_as_windows
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""

    def __init__(
        self,
        obs_shape,
        action_size,
        capacity,
        batch_size,
        device,
        image_size=84,
        pre_image_size=84,
        transform=None,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.pre_image_size = pre_image_size  # for translation
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, action_size), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_cpc(self):
        start = time.time()
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = fast_random_crop(obses, self.image_size)
        next_obses = fast_random_crop(next_obses, self.image_size)
        pos = fast_random_crop(pos, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(
            obs_anchor=obses, obs_pos=pos, time_anchor=None, time_pos=None
        )

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def sample_rad(self, aug_funcs):
        # augs specified as flags
        # curl_sac organizes flags into aug funcs
        # passes aug funcs into sampler

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        if aug_funcs:
            for aug, func in aug_funcs.items():
                # apply crop and cutout first
                if "crop" in aug or "cutout" in aug:
                    obses = func(obses)
                    next_obses = func(next_obses)
                elif "translate" in aug:
                    og_obses = center_crop_images(obses, self.pre_image_size)
                    og_next_obses = center_crop_images(next_obses, self.pre_image_size)
                    obses, rndm_idxs = func(
                        og_obses, self.image_size, return_random_idxs=True
                    )
                    next_obses = func(og_next_obses, self.image_size, **rndm_idxs)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        obses = obses / 255.0
        next_obses = next_obses / 255.0

        # augmentations go here
        if aug_funcs:
            for aug, func in aug_funcs.items():
                # skip crop and cutout augs
                if "crop" in aug or "cutout" in aug or "translate" in aug:
                    continue
                obses = func(obses)
                next_obses = func(next_obses)

        return obses, actions, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, "%d_%d.pt" % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save : self.idx],
            self.next_obses[self.last_save : self.idx],
            self.actions[self.last_save : self.idx],
            self.rewards[self.last_save : self.idx],
            self.not_dones[self.last_save : self.idx],
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split(".")[0].split("_")]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(0, self.capacity if self.full else self.idx, size=1)
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity


class OfflineEpisodeReplayBuffer(SimpleReplayBuffer):
    def __init__(
        self,
        max_replay_buffer_size,
        env,
        max_path_length,
        observation_dim,
        action_dim,
        replace=True,
        batch_length=50,
        use_batch_length=False,
    ):
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        self._observation_dim = get_dim(self._ob_space)
        self._action_dim = get_dim(self._action_space)
        self.max_path_length = max_path_length
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros(
            (max_replay_buffer_size, max_path_length, observation_dim),
            dtype=np.uint8,
        )
        self._actions = np.zeros((max_replay_buffer_size, max_path_length, action_dim))
        self._rewards = np.zeros((max_replay_buffer_size, max_path_length, 1))
        self._terminals = np.zeros(
            (max_replay_buffer_size, max_path_length, 1), dtype="uint8"
        )
        self._replace = replace
        self.batch_length = batch_length
        self.use_batch_length = use_batch_length
        self._top = 0
        self._size = 0

    def add_path(self, path):
        self._observations[self._top : self._top + self.env.n_envs] = path[
            "observations"
        ].transpose(1, 0, 2)
        self._actions[self._top : self._top + self.env.n_envs] = path[
            "actions"
        ].transpose(1, 0, 2)
        self._rewards[self._top : self._top + self.env.n_envs] = np.expand_dims(
            path["rewards"].transpose(1, 0), -1
        )
        self._terminals[self._top : self._top + self.env.n_envs] = np.expand_dims(
            path["terminals"].transpose(1, 0), -1
        )

        self._advance()

    def _advance(self):
        self._top = (self._top + self.env.n_envs) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += self.env.n_envs

    def random_batch(self, batch_size):
        if self.use_batch_length:
            indices = np.random.choice(
                self._size,
                size=batch_size,
                replace=True,
            )
            if not self._replace and self._size < batch_size:
                warnings.warn(
                    "Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay."
                )
            batch_start = np.random.randint(
                0, self.max_path_length - self.batch_length, size=(batch_size)
            )
            batch_indices = np.linspace(
                batch_start,
                batch_start + self.batch_length,
                self.batch_length,
                endpoint=False,
            ).astype(int)

            observations = self._observations[indices][
                np.arange(batch_size), batch_indices
            ].transpose(1, 0, 2)
            actions = self._actions[indices][
                np.arange(batch_size), batch_indices
            ].transpose(1, 0, 2)
            rewards = self._rewards[indices][
                np.arange(batch_size), batch_indices
            ].transpose(1, 0, 2)
            terminals = self._terminals[indices][
                np.arange(batch_size), batch_indices
            ].transpose(1, 0, 2)
        else:
            indices = np.random.choice(
                self._size,
                size=batch_size,
                replace=self._replace or self._size < batch_size,
            )
            if not self._replace and self._size < batch_size:
                warnings.warn(
                    "Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay."
                )
            observations = self._observations[indices]
            actions = self._actions[indices]
            rewards = self._rewards[indices]
            terminals = self._terminals[indices]
        batch = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )
        return batch


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top : top + new_h, left : left + new_w]
    return image


def center_crop_images(image, output_size):
    h, w = image.shape[2:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, :, top : top + new_h, left : left + new_w]
    return image


def center_translate(image, size):
    c, h, w = image.shape
    assert size >= h and size >= w
    outs = np.zeros((c, size, size), dtype=image.dtype)
    h1 = (size - h) // 2
    w1 = (size - w) // 2
    outs[:, h1 : h1 + h, w1 : w1 + w] = image
    return outs


def gaussian_kl_divergence(q_mu, q_logsigma, p_mu, p_logsigma):
    # KL(q, p) where q, p are both Gaussians
    # q_mu, q_logsigma: mean and log std of q
    # p_mu, p_logsigma: same for p
    return (
        p_logsigma
        - q_logsigma
        + (torch.exp(q_logsigma) ** 2 + (q_mu - p_mu) ** 2)
        / (2 * torch.exp(p_logsigma) ** 2)
        - 0.5
    )
