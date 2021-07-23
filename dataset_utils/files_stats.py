import numpy as np
from loguru import logger
import tensorflow as tf

from dataset_utils.image_utils import decode_image
from pathlib import Path

MASK_PATH = Path("C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/masks/test/mask_angular_logo_lettre.png")
IMAGE_PATH = Path("C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/angular_logo.png")


def count_mask_value_occurences(mask_path: Path) -> {int: float}:
    mask_tensor = decode_image(mask_path)
    unique_with_count_tensor = tf.unique_with_counts(tf.reshape(mask_tensor, [-1]))
    values_array = unique_with_count_tensor.y.numpy()
    count_array = unique_with_count_tensor.count.numpy()
    percent_dict = dict(zip(values_array, np.round(count_array / count_array.sum(), decimals=3)))
    # todo : remove the hardcoded values 0 and 255
    logger.info(
        f"\nBackground percent : {percent_dict[0] * 100}"
        f"\nValue percent : {percent_dict[255] * 100}"
    )
    return percent_dict
