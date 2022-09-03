import os
import pandas as pd

image_shape = (768, 768)


ship_dir = 'data/airbus-ship-detection'
train_image_dir = os.path.join(ship_dir, 'train_v2')
test_image_dir = os.path.join(ship_dir, 'test_v2')

masks = pd.read_csv(
    os.path.join(
        'data/airbus-ship-detection/',
        'train_ship_segmentations_v2.csv'))
