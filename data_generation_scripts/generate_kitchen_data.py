import d4rl
import os
from tqdm import tqdm
import torch
import numpy as np
import random
import gym
import pickle
import h5py

imwidth = 128
imheight = 128


def append_data(name, rendered_frames):
    with h5py.File(name, "a") as f:
        f["rendered_frames"].resize(
            (len(rendered_frames) + f["rendered_frames"].shape[0]), axis=0
        )
        f["rendered_frames"][-len(rendered_frames) :] = np.stack(rendered_frames)


def generate_kitchen_data(data_dir):
    env = gym.make("kitchen-mixed-v0")
    dataset = env.get_dataset()
    obs, acs, dones, rewards = (
        dataset["observations"],
        dataset["actions"],
        dataset["terminals"],
        dataset["rewards"],
    )
    seq_end_idxs = np.where(dones)[0]
    rendered_frames = np.empty((0, imwidth, imheight, 3), dtype=np.uint8)
    data_name = os.path.join(data_dir, "kitchen_mixed_data.h5")
    # creat new dataset
    with h5py.File(data_name, "w") as f:
        # first make a key for each of the keys in the original dataset
        for key in dataset.keys():
            f.create_dataset(key, data=dataset[key])
        # now add the rendered frames
        f.create_dataset(
            "rendered_frames",
            data=np.array(rendered_frames),
            maxshape=(None, imwidth, imheight, 3),
            compression="gzip",
        )
    rendered_frames = []
    print(f"len(obs): {len(obs)}, len(acs): {len(acs)}")
    for i in tqdm(range(len(acs))):
        ob, ac = obs[i], acs[i]
        env.reset_model_state(ob)
        test_ob = env.env.env._get_obs()
        assert np.allclose(ob, test_ob, atol=0.1)
        rendered_frames.append(
            env.render(mode="rgb_array", imwidth=imwidth, imheight=imheight)
        )
        # every 1k frames let's save to the dataset
        if i % 500 == 0 or i == len(acs) - 1:
            append_data(data_name, rendered_frames)
            rendered_frames = []


if __name__ == "__main__":
    generate_kitchen_data("./")
