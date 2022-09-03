from constants import ship_dir, train_image_dir, test_image_dir, masks
from utils import dice_coef, rle_encode, rle_decode, multi_rle_encode, masks_as_color, masks_as_image
import os
import random
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
gc.enable()  # memory is tight


IMG_SIZE = 128
MODEL_PATH = "model.h5"


def montage_rgb(x): return np.stack(
    [montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)


# predict and visualize
def raw_prediction(img, path=test_image_dir):
    c_img = imread(os.path.join(path, c_img_name))
    c_img = resize(c_img, (IMG_SIZE, IMG_SIZE), mode='constant', preserve_range=True)
    c_img = np.expand_dims(c_img, 0) / 255.0
    cur_seg = model.predict(c_img)[0]
    return cur_seg, c_img[0]


def smooth(cur_seg):
    return binary_opening(cur_seg > 0.99, np.expand_dims(disk(2), -1))


def predict(img, path=test_image_dir):
    cur_seg, c_img = raw_prediction(img, path=path)
    return smooth(cur_seg), c_img


# PREPROCESSIN
not_empty = pd.notna(masks.EncodedPixels)
print(
    not_empty.sum(),
    'masks in',
    masks[not_empty].ImageId.nunique(),
    'images')
print((~not_empty).sum(), 'empty images in',
      masks.ImageId.nunique(), 'total images')
masks.head()


masks['ships'] = masks['EncodedPixels'].map(
    lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(
    lambda x: 1.0 if x > 0 else 0.0)
unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
# some files are too small/corrupt
unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(
    lambda c_img_id: os.stat(os.path.join(train_image_dir, c_img_id)).st_size / 1024)

# keep only +50kb files
unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > 50]
masks.drop(['ships'], axis=1, inplace=True)
unique_img_ids.sample(7)


# ships in group
SAMPLES_PER_GROUP = 2000
balanced_train_df = unique_img_ids.groupby('ships').apply(
    lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)
print(balanced_train_df.shape[0], 'masks')

# split data
train_ids, valid_ids = train_test_split(balanced_train_df,
                                        test_size=0.2,
                                        stratify=balanced_train_df['ships'])
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')

# load pretrained model
model = load_model(MODEL_PATH,
                   custom_objects={'dice_coef': dice_coef})

# compare 4 img
samples = valid_df.groupby('ships').apply(lambda x: x.sample(1))
fig, m_axs = plt.subplots(4, 4, figsize=(30, 30))
[c_ax.axis('off') for c_ax in m_axs.flatten()]

for (ax1, ax2, ax3, ax4), c_img_name in zip(m_axs, samples.ImageId.values):
    first_seg, first_img = raw_prediction(c_img_name, train_image_dir)
    ax1.imshow(first_img)
    ax1.set_title('Image: ' + c_img_name)
    ax2.imshow(first_seg[:, :, 0], cmap=get_cmap('jet'))
    ax2.set_title('Model Prediction')
    reencoded = masks_as_color(multi_rle_encode(smooth(first_seg)[:, :, 0]))
    ax3.imshow(reencoded)
    ax3.set_title('Prediction Masks')
    ground_truth = masks_as_color(masks.query(
        'ImageId=="{}"'.format(c_img_name))['EncodedPixels'])
    ax4.imshow(ground_truth)
    ax4.set_title('Ground Truth')

plt.show()
