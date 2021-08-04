from collections import Counter

import numpy as np
import tensorflow as tf
import os

from dataset_utils.image_utils import (
    decode_image,
    get_images_paths,
    get_file_name_with_extension,
)
from pathlib import Path

MASK_PATH = Path(
    "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/masks/test/mask_angular_logo_lettre.png"
)
IMAGE_PATH = Path(
    "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/1/1.png"
)
MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/all")
CATEGORICAL_MASKS_DIR = Path(
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/categorical_masks"
)
IMAGES_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images")
IMAGES_DIR = Path("C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/sorted_images/kept/Marie_images")


def count_mask_value_occurences(mask_path: Path) -> {int: float}:
    mask_tensor = decode_image(mask_path)
    unique_with_count_tensor = tf.unique_with_counts(tf.reshape(mask_tensor[:, :, 0], [-1]))
    values_array = unique_with_count_tensor.y.numpy()
    count_array = unique_with_count_tensor.count.numpy()
    count_dict = dict(
        zip(values_array, count_array)
    )
    return count_dict


def count_mask_value_occurences_of_2d_tensor(tensor: tf.Tensor) -> {int: float}:
    unique_with_count_tensor = tf.unique_with_counts(tf.reshape(tensor, [-1]))
    values_array = unique_with_count_tensor.y.numpy()
    count_array = unique_with_count_tensor.count.numpy()
    count_dict = dict(
        zip(values_array, count_array)
    )
    return count_dict


def count_mask_value_occurences_percent(mask_path: Path) -> {int: float}:
    mask_tensor = decode_image(mask_path)
    unique_with_count_tensor = tf.unique_with_counts(tf.reshape(mask_tensor, [-1]))
    values_array = unique_with_count_tensor.y.numpy()
    count_array = unique_with_count_tensor.count.numpy()
    percent_dict = dict(
        zip(values_array, np.round(count_array / count_array.sum(), decimals=3))
    )
    return percent_dict


def count_label_masks_dirs(masks_dir: Path):
    return len(list(masks_dir.iterdir()))


def get_file_size(file_path: Path):
    """Return the number of bytes."""
    return os.path.getsize(str(file_path))


def get_all_images_size(images_dir: Path):
    image_sizes = [
        os.path.getsize(str(image_path))
        for image_sub_dir in list(images_dir.iterdir())
        for image_path in list(image_sub_dir.iterdir())
    ]
    return sum(image_sizes)


def get_all_categorical_masks_size(categorical_masks_dir: Path):
    categorical_masks_sizes = [
        os.path.getsize(str(image_path))
        for image_sub_dir in list(categorical_masks_dir.iterdir())
        for image_path in list(image_sub_dir.iterdir())
    ]
    return sum(categorical_masks_sizes)


def get_and_count_images_shapes(images_dir: Path):
    images_shapes = [
        tuple(decode_image(image_path).shape)
        for image_path in get_images_paths(images_dir)
    ]
    return dict(Counter(images_shapes))


def get_images_with_shape_different_than(shape: tuple, images_dir: Path):
    return {
        get_file_name_with_extension(image_path): tuple(decode_image(image_path).shape)
        for image_path in get_images_paths(images_dir)
        if tuple(decode_image(image_path).shape) != shape
    }


def get_image_shape(image_path: Path) -> tuple:
    return tuple(decode_image(image_path).shape)

# get_images_with_shape_different_than((2848, 4288, 3), IMAGES_DIR)
# dict(Counter([tuple(decode_image(image_path).shape) for image_path in [image_path for image_path in IMAGES_DIR.iterdir()]]))
