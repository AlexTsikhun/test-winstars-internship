import pathlib
import os
import tensorflow as tf
from keras.models import load_model
import keras.backend as K
from skimage.io import imread, imshow
from skimage.transform import resize

from sklearn.model_selection import train_test_split

from matplotlib.cm import get_cmap
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from skimage.morphology import binary_opening, disk, label
import gc
import segmentation_models as sm

gc.enable()  # memory is tight

from constants import (
    train_image_dir,
    test_image_dir,
    masks,
    IMG_SIZE,
    SAMPLES_PER_GROUP,
)
from utils import dice_coef, multi_rle_encode, masks_as_color
from preprocessing import preprocess_data


def raw_prediction(img, path=test_image_dir):
    c_img = imread(os.path.join(path, c_img_name))
    c_img = resize(c_img, (IMG_SIZE, IMG_SIZE), mode="constant", preserve_range=True)
    c_img = np.expand_dims(c_img, 0) / 255.0
    cur_seg = model.predict(c_img)[0]
    return cur_seg, c_img[0]


def smooth(cur_seg):
    return binary_opening(cur_seg > 0.99, np.expand_dims(disk(2), -1))


MODEL_PATH = pathlib.Path.cwd().parents[0].joinpath("models/unet_model.h5")

balanced_train_df = preprocess_data(train_image_dir, masks, SAMPLES_PER_GROUP)

train_ids, valid_ids = train_test_split(
    balanced_train_df, test_size=0.2, stratify=balanced_train_df["ships"]
)
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0], "training masks")
print(valid_df.shape[0], "validation masks")

model = load_model(
    MODEL_PATH,
    custom_objects={
        "dice_coef": dice_coef,
        "binary_crossentropy_plus_jaccard_loss": sm.losses.bce_jaccard_loss,
    },
)

# compare 4 img
samples = valid_df.groupby("ships").apply(lambda x: x.sample(1))

fig, m_axs = plt.subplots(4, 4, figsize=(30, 30))
[c_ax.axis("off") for c_ax in m_axs.flatten()]

for (ax1, ax2, ax3, ax4), c_img_name in zip(m_axs, samples.ImageId.values):
    first_seg, first_img = raw_prediction(c_img_name, train_image_dir)
    ax1.imshow(first_img)
    ax1.set_title("Image: " + c_img_name)
    ax2.imshow(first_seg[:, :, 0], cmap=get_cmap("jet"))
    ax2.set_title("Model Prediction")
    reencoded = masks_as_color(multi_rle_encode(smooth(first_seg)[:, :, 0]))
    ax3.imshow(reencoded)
    ax3.set_title("Prediction Masks")
    ground_truth = masks_as_color(
        masks.query('ImageId=="{}"'.format(c_img_name))["EncodedPixels"]
    )
    ax4.imshow(ground_truth)
    ax4.set_title("Ground Truth")

plt.show()
