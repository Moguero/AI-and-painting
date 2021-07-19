import time
from loguru import logger
from label_utils.mask_loader import get_image_masks_first_channel, load_mask
import numpy as np
import pandas as pd

IMAGE_PATH = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043/_DSC0043.JPG"
MASKS_DIR = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/"



def stack_image_masks(image_path: str, masks_dir: str):
    """Creates a unique one-hot encoded mask by stacking the image masks."""
    image_masks = get_image_masks_first_channel(image_path, masks_dir)
    one_hot_encoded_masks = [one_hot_encode_mask(mask) for mask in image_masks]
    stacked_mask = np.stack(one_hot_encoded_masks, axis=2)
    return stacked_mask


# todo : implement the one-encoding algorithm with tensorflow
def one_hot_encode_mask(mask: np.ndarray):
    return np.where(mask == 255, 1, 0)


# todo : delete this
def transform_png_image_ndarray_to_dataframe(mask_path: str):
    # transform png to numpy array to dataframe
    start_time = time.time()
    logger.info("Start the conversion from PNG to DataFrame...")
    img = load_mask(mask_path)
    multi_index = pd.MultiIndex.from_product([range(s) for s in img.shape], names=["x", "y", "channel"])
    img_df = pd.DataFrame({"img": img.flatten()}, index=multi_index).unstack("channel").rename(columns={0: "R", 1: "G", 2: "B", 3:"A"})

    # save dataframe for future use
    img_df.to_pickle()
    logger.info(f"\nConversion from PNG to DataFrame took : {round(time.time() - start_time, 3)} seconds to execute.")  # usual order of magnitude : 20s
    return img_df


# todo : transform all the numpy array to dataframe directly
