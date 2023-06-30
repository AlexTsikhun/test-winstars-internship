import keras.backend as K
from skimage.morphology import binary_opening, disk, label
import numpy as np
import pandas as pd
from constants import image_shape

# from winstars-model-tr import image_shape


def dice_coef(y_true, y_pred, smooth=1):
    """
    Calculate the dice coefficient between the ground truth and predicted segmentation masks.

    Args:
        y_true (tensor): Ground truth segmentation mask.
        y_pred (tensor): Predicted segmentation mask.
        smooth (float, optional): Smoothing factor. Defaults to 1.

    Returns:
        float: Dice coefficient between the ground truth and predicted masks.
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    numerator = 2.0 * intersection + smooth
    denominator = K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth
    dice_coef = numerator / denominator
    return dice_coef

def multi_rle_encode(img, **kwargs):
    """
    Encode connected regions as separated masks
    """
    labels = label(img)
    if img.ndim > 2:
        # For image with >2 dimestions
        return [
            rle_encode(np.sum(labels == k, axis=2), **kwargs)
            for k in np.unique(labels[labels > 0])
        ]
    else:
        # For grayscale
        return [
            rle_encode(labels == k, **kwargs) for k in np.unique(labels[labels > 0])
        ]

def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    if np.max(img) < min_max_threshold:
        return ""  # no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return ""  # ignore overfilled mask
    # Flatten the image and add padding zeros at the beginning and end
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    # Find the starting and ending positions of each run
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=image_shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if pd.isnull(mask_rle):
        img = no_mask
        return img.reshape(shape).T
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all
    # ships
    all_masks = np.zeros(image_shape, dtype=np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each
    # ships
    all_masks = np.zeros(image_shape, dtype=np.float64)

    def scale(x):
        return (len(in_mask_list) + x + 1) / (
            len(in_mask_list) * 2
        )  # scale the heatmap image to shift

    for i, mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:, :] += scale(i) * rle_decode(mask)
    return all_masks

no_mask = np.zeros(image_shape[0] * image_shape[1], dtype=np.uint8)
