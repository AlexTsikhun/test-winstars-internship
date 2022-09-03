#!/usr/bin/env python
# coding: utf-8

import os
import random
import tensorflow as tf
from keras.models import load_model
import keras.backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize

from sklearn.model_selection import train_test_split

from matplotlib.cm import get_cmap

import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from skimage.segmentation import mark_boundaries
# from skimage.io import montage2d as montage
from skimage.morphology import binary_opening, disk, label
import gc
gc.enable()  # memory is tight


IMG_WIDTH = 768
IMG_HEIGHT = 768
IMG_CHANNELS = 3
TARGET_WIDTH = 128
TARGET_HEIGHT = 128
epochs = 10
batch_size = 32
image_shape = (768, 768)
FAST_RUN = True  # use for development only
FAST_PREDICTION = True  # use for development only
MAX_TRAIN_STEPS = 9
MAX_TRAIN_EPOCHS = 99


def montage_rgb(x): return np.stack(
    [montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)


ship_dir = 'data/airbus-ship-detection'
train_image_dir = os.path.join(ship_dir, 'train_v2')
test_image_dir = os.path.join(ship_dir, 'test_v2')
MODEL_PATH = "model.h5"

IMG_SIZE = 128


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / \
        (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels == k, axis=2), **kwargs)
                for k in np.unique(labels[labels > 0])]
    else:
        return [rle_encode(labels == k, **kwargs)
                for k in np.unique(labels[labels > 0])]


def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return ''  # no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return ''  # ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all
    # ships
    all_masks = np.zeros((768, 768), dtype=np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each
    # ships
    all_masks = np.zeros((768, 768), dtype=np.float64)
    def scale(x): return (len(in_mask_list) + x + 1) / \
        (len(in_mask_list) * 2)  # scale the heatmap image to shift
    for i, mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:, :] += scale(i) * rle_decode(mask)
    return all_masks


# predict and visualize
def raw_prediction(img, path=test_image_dir):
    c_img = imread(os.path.join(path, c_img_name))
    c_img = resize(c_img, (128, 128), mode='constant', preserve_range=True)
    c_img = np.expand_dims(c_img, 0) / 255.0
    cur_seg = model.predict(c_img)[0]
    return cur_seg, c_img[0]


def smooth(cur_seg):
    return binary_opening(cur_seg > 0.99, np.expand_dims(disk(2), -1))


def predict(img, path=test_image_dir):
    cur_seg, c_img = raw_prediction(img, path=path)
    return smooth(cur_seg), c_img


masks = pd.read_csv(
    os.path.join(
        'data/airbus-ship-detection/',
        'train_ship_segmentations_v2.csv'))
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
# unique_img_ids['file_size_kb'].hist()
masks.drop(['ships'], axis=1, inplace=True)
unique_img_ids.sample(7)


SAMPLES_PER_GROUP = 2000
balanced_train_df = unique_img_ids.groupby('ships').apply(
    lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)
# balanced_train_df['ships'].hist(bins=balanced_train_df['ships'].max()+1)
print(balanced_train_df.shape[0], 'masks')


train_ids, valid_ids = train_test_split(balanced_train_df,
                                        test_size=0.2,
                                        stratify=balanced_train_df['ships'])
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')


model = load_model(MODEL_PATH,
                   custom_objects={'dice_coef': dice_coef})


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
