import matplotlib.pyplot as plt
import pathlib
from skimage.transform import resize
import os
from skimage.io import imread
from matplotlib.cm import get_cmap

from constants import IMG_WIDTH, IMG_HEIGHT, image_shape

def plot_images(df_sub, predict_mask):

    fig = plt.figure(figsize=(16, 8))
    for index, row in df_sub.head(20).iterrows():
        origin_image = imread(
            pathlib.Path.cwd()
            .parents[0]
            .joinpath(os.path.join("data/test_v2/"), row.ImageId)
        )
        predicted_image = (
            resize(predict_mask[index], image_shape).reshape(IMG_WIDTH, IMG_HEIGHT) * 255
        )
        plt.subplot(10, 4, 2 * index + 1)
        plt.imshow(origin_image)
        plt.subplot(10, 4, 2 * index + 2)
        plt.imshow(predicted_image)


def plot_training_metrics(history):
    plt.figure(figsize=(15, 5))
    plt.plot(
        range(history.epoch[-1] + 1),
        history.history["val_dice_coef"],
        label="Val_dice_coef",
    )
    plt.plot(
        range(history.epoch[-1] + 1), history.history["dice_coef"], label="Trn_dice_coef"
    )
    plt.title("DICE")
    plt.xlabel("Epoch")
    plt.ylabel("dice_coef")
    plt.legend()
    plt.show()
