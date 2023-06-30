import pathlib
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tensorflow.keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import warnings
import numpy as np
import pandas as pd
import argparse

from skimage.io import imread
import matplotlib.pyplot as plt

# from skimage.io import montage2d as montage
import gc

gc.enable()  # memory is tight

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore")

from model import unet_model
from utils import rle_decode
from constants import *
from visualization import plot_images, plot_training_metrics
from preprocessing import preprocess_data

def get_image(image_name):
    img = imread(
        pathlib.Path.cwd()
        .parents[0]
        .joinpath(os.path.join("data/train_v2/"), image_name)
    )[:, :, :IMG_CHANNELS]
    img = resize(
        img, (TARGET_WIDTH, TARGET_HEIGHT), mode="constant", preserve_range=True
    )
    return img


def get_mask(code):
    img = rle_decode(code)
    img = resize(
        img, (TARGET_WIDTH, TARGET_HEIGHT, 1), mode="constant", preserve_range=True
    )
    return img


def create_image_generator(precess_batch_size, data_df):
    while True:
        for k, group_df in data_df.groupby(
            np.arange(data_df.shape[0]) // precess_batch_size
        ):
            imgs = []
            labels = []
            for index, row in group_df.iterrows():
                # images
                original_img = get_image(row.ImageId) / 255.0
                # masks
                mask = get_mask(row.EncodedPixels) / 255.0

                imgs.append(original_img)
                labels.append(mask)

            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels


def get_test_image(image_name):
    img = imread(
        pathlib.Path.cwd()
        .parents[0]
        .joinpath(os.path.join("data/test_v2/"), image_name)
    )[:, :, :IMG_CHANNELS]
    img = resize(
        img, (TARGET_WIDTH, TARGET_HEIGHT), mode="constant", preserve_range=True
    )
    return img


def create_test_generator(precess_batch_size):
    while True:
        for k, ix in df_sub.groupby(np.arange(df_sub.shape[0]) // precess_batch_size):
            imgs = []
            labels = []
            for index, row in ix.iterrows():
                original_img = get_test_image(row.ImageId) / 255.0
                imgs.append(original_img)

            imgs = np.array(imgs)
            yield imgs


# ARGPARSER
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--train_epoch", type=int, default=99)
parser.add_argument("-s", "--train_steps", type=int, default=50)
parser.add_argument("-bs", "--batch_size", type=int, default=32)

args = parser.parse_args()

# pathlib.Path(__file__).parents[1].joinpath(...) - back step  in my catalog
df_sub = pd.read_csv(
    pathlib.Path.cwd().parents[0].joinpath("data/sample_submission_v2.csv")
)
print('1')
balanced_train_df = preprocess_data(train_image_dir, masks, SAMPLES_PER_GROUP)

print('2')

# split data
train_ids, valid_ids = train_test_split(
    balanced_train_df, test_size=0.2, stratify=balanced_train_df["ships"]
)

train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0], "training masks")
print(valid_df.shape[0], "validation masks")

model = unet_model()
print('3')

train_generator = create_image_generator(args.batch_size, train_df)
validate_generator = create_image_generator(args.batch_size, valid_df)
print('4')

# # Train
# Save best model at every epoch
checkpoint = ModelCheckpoint(
    "models/unet_model.h5",
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
)

print('5')

train_steps = min(args.train_steps, train_df.shape[0] // args.batch_size)
validate_steps = min(args.train_steps, valid_df.shape[0] // args.batch_size)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=validate_generator,
    callbacks=[checkpoint],
    validation_steps=validate_steps,
    epochs=args.train_epoch,
)
print('')

test_generator = create_test_generator(args.batch_size)

# # Predict
test_steps = np.ceil(float(df_sub.shape[0]) / float(args.batch_size)).astype(int)
predict_mask = model.predict_generator(test_generator, steps=test_steps)
print('plot')
plot_images(df_sub, predict_mask, history)
plot_training_metrics(history)

predict_mask.shape
