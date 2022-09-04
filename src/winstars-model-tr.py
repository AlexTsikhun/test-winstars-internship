import pathlib
from utils import rle_decode, masks_as_image, masks_as_color, dice_coef
from constants import masks, ship_dir, train_image_dir, test_image_dir, image_shape
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
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
from matplotlib.cm import get_cmap
from skimage.segmentation import mark_boundaries
# from skimage.io import montage2d as montage
from skimage.morphology import binary_opening, disk, label
import gc
gc.enable()  # memory is tight

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings('ignore')


IMG_WIDTH = 768
IMG_HEIGHT = 768
IMG_CHANNELS = 3
TARGET_WIDTH = 128
TARGET_HEIGHT = 128

SAMPLES_PER_GROUP = 2000


def get_image(image_name):
    img = imread(
        pathlib.Path(__file__).parents[1].joinpath(
            os.path.join('data/airbus-ship-detection/train_v2/'),
            image_name))[
        :,
        :,
        :IMG_CHANNELS]
    img = resize(img, (TARGET_WIDTH, TARGET_HEIGHT),
                 mode='constant', preserve_range=True)
    return img


def get_mask(code):
    img = rle_decode(code)
    img = resize(img, (TARGET_WIDTH, TARGET_HEIGHT, 1),
                 mode='constant', preserve_range=True)
    return img


def create_image_generator(precess_batch_size, data_df):
    while True:
        for k, group_df in data_df.groupby(
                np.arange(data_df.shape[0]) // precess_batch_size):
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
        pathlib.Path(__file__).parents[1].joinpath(
            os.path.join('data/airbus-ship-detection/test_v2/'),
            image_name))[
        :,
        :,
        :IMG_CHANNELS]
    img = resize(img, (TARGET_WIDTH, TARGET_HEIGHT),
                 mode='constant', preserve_range=True)
    return img


def create_test_generator(precess_batch_size):
    while True:
        for k, ix in df_sub.groupby(
                np.arange(df_sub.shape[0]) // precess_batch_size):
            imgs = []
            labels = []
            for index, row in ix.iterrows():
                original_img = get_test_image(row.ImageId) / 255.0
                imgs.append(original_img)

            imgs = np.array(imgs)
            yield imgs

######

def montage_rgb(x): return np.stack(
    [montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)


# ARGPARSER
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--train_epoch', type=int, default=99)
parser.add_argument('-s', '--train_steps', type=int, default=50)
parser.add_argument('-bs', '--batch_size', type=int, default=32)

args = parser.parse_args()


# pathlib.Path(__file__).parents[1].joinpath(...) - back step  in my catalog
df_sub = pd.read_csv(pathlib.Path(__file__).parents[1].joinpath(
    'data/airbus-ship-detection/sample_submission_v2.csv'))


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
unique_img_ids['file_size_kb'].hist()
masks.drop(['ships'], axis=1, inplace=True)
unique_img_ids.sample(7)


# ships in group
balanced_train_df = unique_img_ids.groupby('ships').apply(
    lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)
balanced_train_df['ships'].hist(bins=balanced_train_df['ships'].max() + 1)
print(balanced_train_df.shape[0], 'masks')

# split data
train_ids, valid_ids = train_test_split(balanced_train_df,
                                        test_size=0.2,
                                        stratify=balanced_train_df['ships'])
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')


# Model interface
inputs = Input((TARGET_WIDTH, TARGET_HEIGHT, IMG_CHANNELS))

# 128

down1 = Conv2D(64, (3, 3), padding='same')(inputs)
down1 = BatchNormalization()(down1)
down1 = Activation('relu')(down1)
down1 = Conv2D(64, (3, 3), padding='same')(down1)
down1 = BatchNormalization()(down1)
down1 = Activation('relu')(down1)
down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
# 64

down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
down2 = BatchNormalization()(down2)
down2 = Activation('relu')(down2)
down2 = Conv2D(128, (3, 3), padding='same')(down2)
down2 = BatchNormalization()(down2)
down2 = Activation('relu')(down2)
down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
# 32
down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
down3 = BatchNormalization()(down3)
down3 = Activation('relu')(down3)
down3 = Conv2D(256, (3, 3), padding='same')(down3)
down3 = BatchNormalization()(down3)
down3 = Activation('relu')(down3)
down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
# 16
down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
down4 = BatchNormalization()(down4)
down4 = Activation('relu')(down4)
down4 = Conv2D(512, (3, 3), padding='same')(down4)
down4 = BatchNormalization()(down4)
down4 = Activation('relu')(down4)
down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
# 8
center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
center = BatchNormalization()(center)
center = Activation('relu')(center)
center = Conv2D(1024, (3, 3), padding='same')(center)
center = BatchNormalization()(center)
center = Activation('relu')(center)
# center
up4 = UpSampling2D((2, 2))(center)
up4 = concatenate([down4, up4], axis=3)
up4 = Conv2D(512, (3, 3), padding='same')(up4)
up4 = BatchNormalization()(up4)
up4 = Activation('relu')(up4)
up4 = Conv2D(512, (3, 3), padding='same')(up4)
up4 = BatchNormalization()(up4)
up4 = Activation('relu')(up4)
up4 = Conv2D(512, (3, 3), padding='same')(up4)
up4 = BatchNormalization()(up4)
up4 = Activation('relu')(up4)
# 16
up3 = UpSampling2D((2, 2))(up4)
up3 = concatenate([down3, up3], axis=3)
up3 = Conv2D(256, (3, 3), padding='same')(up3)
up3 = BatchNormalization()(up3)
up3 = Activation('relu')(up3)
up3 = Conv2D(256, (3, 3), padding='same')(up3)
up3 = BatchNormalization()(up3)
up3 = Activation('relu')(up3)
up3 = Conv2D(256, (3, 3), padding='same')(up3)
up3 = BatchNormalization()(up3)
up3 = Activation('relu')(up3)
# 32
up2 = UpSampling2D((2, 2))(up3)
up2 = concatenate([down2, up2], axis=3)
up2 = Conv2D(128, (3, 3), padding='same')(up2)
up2 = BatchNormalization()(up2)
up2 = Activation('relu')(up2)
up2 = Conv2D(128, (3, 3), padding='same')(up2)
up2 = BatchNormalization()(up2)
up2 = Activation('relu')(up2)
up2 = Conv2D(128, (3, 3), padding='same')(up2)
up2 = BatchNormalization()(up2)
up2 = Activation('relu')(up2)
# 64
up1 = UpSampling2D((2, 2))(up2)
up1 = concatenate([down1, up1], axis=3)
up1 = Conv2D(64, (3, 3), padding='same')(up1)
up1 = BatchNormalization()(up1)
up1 = Activation('relu')(up1)
up1 = Conv2D(64, (3, 3), padding='same')(up1)
up1 = BatchNormalization()(up1)
up1 = Activation('relu')(up1)
up1 = Conv2D(64, (3, 3), padding='same')(up1)
up1 = BatchNormalization()(up1)
up1 = Activation('relu')(up1)
# 128
outputs = Conv2D(1, (1, 1), activation='sigmoid')(up1)

model = Model(inputs=inputs, outputs=outputs)

optimizer = tf.keras.optimizers.RMSprop(1e-4)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=[dice_coef]
)


train_generator = create_image_generator(args.batch_size, train_df)
validate_generator = create_image_generator(args.batch_size, valid_df)

# # Train
# Save best model at every epoch
checkpoint = ModelCheckpoint(
    'model.h5',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'
)


train_steps = min(args.train_steps, train_df.shape[0] // args.batch_size)
validate_steps = min(args.train_steps, valid_df.shape[0] // args.batch_size)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=validate_generator,
    callbacks=[checkpoint],
    validation_steps=validate_steps,
    epochs=args.train_epoch
)


test_generator = create_test_generator(args.batch_size)

# # Predict
test_steps = np.ceil(float(df_sub.shape[0]) / float(args.batch_size)).astype(int)
predict_mask = model.predict_generator(test_generator, steps=test_steps)


fig = plt.figure(figsize=(16, 8))
for index, row in df_sub.head(20).iterrows():
    origin_image = imread(pathlib.Path(__file__).parents[1].joinpath(
        os.path.join('data/airbus-ship-detection/test_v2/'), row.ImageId))
    predicted_image = resize(
        predict_mask[index],
        image_shape).reshape(
        IMG_WIDTH,
        IMG_HEIGHT) * 255
    plt.subplot(10, 4, 2 * index + 1)
    plt.imshow(origin_image)
    plt.subplot(10, 4, 2 * index + 2)
    plt.imshow(predicted_image)

predict_mask.shape

# PLOT TRAINING
plt.figure(figsize=(15, 5))
plt.plot(range(history.epoch[-1] + 1),
         history.history['val_dice_coef'],
         label='Val_dice_coef')
plt.plot(range(history.epoch[-1] + 1),
         history.history['dice_coef'],
         label='Trn_dice_coef')
plt.title('DICE')
plt.xlabel('Epoch')
plt.ylabel('dice_coef')
plt.legend()
plt.show()
