import os
import pandas as pd
import pathlib

IMG_WIDTH = 768
IMG_HEIGHT = 768
IMG_CHANNELS = 3
TARGET_WIDTH = 128
TARGET_HEIGHT = 128
SAMPLES_PER_GROUP = 2000
IMG_SIZE = 128

image_shape = (768, 768)

# Set default path
ship_dir = pathlib.Path.cwd().parents[0].joinpath("data/")

train_image_dir = os.path.join(ship_dir, "train_v2")
test_image_dir = os.path.join(ship_dir, "test_v2")

masks = pd.read_csv(
    pathlib.Path.cwd()
    .parents[0]
    .joinpath(os.path.join("data/", "train_ship_segmentations_v2.csv"))
)
