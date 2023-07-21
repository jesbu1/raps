# this script will load the first 200 frames and turn it into a movie to be saved to a gif
import h5py
import os
import numpy as np
import cv2

data_dir = "./kitchen_mixed_data.h5"
with h5py.File(data_dir, "r") as f:
    frames = f["rendered_frames"][:1000]
    print(f"frames.shape: {frames.shape}")
    # now save the frames as a gif
    # https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
    import imageio

    imageio.mimsave("kitchen_mixed_data.gif", frames, duration=2000)


# now save as sequence of images in a folder
# folder = "test"
# os.makedirs(folder, exist_ok=True)
# for i, frame in enumerate(frames):
#     corrected_color_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(f"{folder}/kitchen_mixed_data_{i}.png", corrected_color_frame)
