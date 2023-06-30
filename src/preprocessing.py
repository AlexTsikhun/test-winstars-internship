import pandas as pd
import os
import pandas as pd
from pathlib import Path
import os
import pandas as pd
from pathlib import Path
import os

def preprocess_data(train_image_dir, masks, SAMPLES_PER_GROUP):
    not_empty = pd.notna(masks.EncodedPixels)
    print(not_empty.sum(), "masks in", masks[not_empty].ImageId.nunique(), "images")
    print((~not_empty).sum(), "empty images in", masks.ImageId.nunique(), "total images")
    masks["ships"] = masks["EncodedPixels"].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = masks.groupby("ImageId").agg({"ships": "sum"}).reset_index()
    unique_img_ids["has_ship"] = unique_img_ids["ships"].map(lambda x: 1.0 if x > 0 else 0.0)
    unique_img_ids["has_ship_vec"] = unique_img_ids["has_ship"].map(lambda x: [x])
    unique_img_ids["file_size_kb"] = unique_img_ids["ImageId"].map(lambda c_img_id: os.stat(os.path.join(train_image_dir, c_img_id)).st_size / 1024)
    unique_img_ids = unique_img_ids[unique_img_ids["file_size_kb"] > 50]
    # unique_img_ids["file_size_kb"].hist()
    masks.drop(["ships"], axis=1, inplace=True)
    unique_img_ids.sample(7)
    balanced_train_df = unique_img_ids.groupby("ships").apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)
    # balanced_train_df["ships"].hist(bins=balanced_train_df["ships"].max() + 1)
    print(balanced_train_df.shape[0], "masks")
    return balanced_train_df
