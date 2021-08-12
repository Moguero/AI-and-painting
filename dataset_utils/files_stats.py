import ast
from collections import Counter

import numpy as np
import tensorflow as tf
import os

from tqdm import tqdm

from dataset_utils.file_utils import save_dict_to_csv, load_saved_dict
from dataset_utils.image_utils import (
    decode_image,
    get_images_paths,
    get_file_name_with_extension,
)
from pathlib import Path

from dataset_utils.masks_encoder import stack_image_patch_masks, stack_image_masks

MASK_PATH = Path(
    "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/masks/test/mask_angular_logo_lettre.png"
)
IMAGE_PATH = Path(
    "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/1/1.png"
)
MASKS_DIR = Path(
    "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/files/labels_masks/all"
)
CATEGORICAL_MASKS_DIR = Path(
    "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/files/categorical_masks"
)
IMAGES_DIR = Path(
    "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/files/images"
)
PATCHES_DIR = Path(r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\patches")
PATCH_PATH = Path(
    r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\patches\1\1\image\patch_1.jpg"
)
COUNT_ALL_CATEGORICAL_MASK_IRREGULAR_PIXELS_OUTPUT_PATH = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files\temp_files\categorical_mask_irregular_pixels_count.csv"
)
INPUT_SHAPE = 256
N_CLASSES = 9


def count_mask_value_occurences(mask_path: Path) -> {int: float}:
    mask_tensor = decode_image(mask_path)
    unique_with_count_tensor = tf.unique_with_counts(
        tf.reshape(mask_tensor[:, :, 0], [-1])
    )
    values_array = unique_with_count_tensor.y.numpy()
    count_array = unique_with_count_tensor.count.numpy()
    count_dict = dict(zip(values_array, count_array))
    return count_dict


def count_mask_value_occurences_of_2d_tensor(tensor: tf.Tensor) -> {int: float}:
    unique_with_count_tensor = tf.unique_with_counts(tf.reshape(tensor, [-1]))
    values_array = unique_with_count_tensor.y.numpy()
    count_array = unique_with_count_tensor.count.numpy()
    count_dict = dict(zip(values_array, count_array))
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


# todo : test all those 4 functions


def count_mask_value_occurences_percent_of_2d_tensor(tensor: tf.Tensor) -> {int: float}:
    unique_with_count_tensor = tf.unique_with_counts(tf.reshape(tensor, [-1]))
    values_array = unique_with_count_tensor.y.numpy()
    count_array = unique_with_count_tensor.count.numpy()
    percent_dict = dict(
        zip(values_array, np.round(count_array / count_array.sum() * 100, decimals=3))
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


def count_total_number_of_patches(patches_dir: Path) -> int:
    return sum([len(list(image_dir.iterdir())) for image_dir in patches_dir.iterdir()])


def count_mask_value_occurences_of_categorical_mask(
    image_path: Path, masks_dir: Path
) -> {int: float}:
    return count_mask_value_occurences_of_2d_tensor(
        stack_image_masks(image_path, masks_dir)
    )


def count_categorical_mask_irregular_pixels(
    image_path: Path, masks_dir: Path, n_classes
) -> {int: int}:
    mask_value_occurences_of_categorical_mask_dict = (
        count_mask_value_occurences_of_categorical_mask(image_path, masks_dir)
    )
    irregular_pixels_dict = {
        key: value
        for key, value in mask_value_occurences_of_categorical_mask_dict.items()
        if key > n_classes
    }
    return irregular_pixels_dict


def save_count_all_categorical_mask_irregular_pixels(
    images_dir_path: Path, masks_dir: Path, n_classes: int, output_path: Path
) -> None:
    dict_to_save = {
        image_path: count_categorical_mask_irregular_pixels(
            image_path, masks_dir, n_classes
        )
        for image_path in tqdm(
            get_images_paths(images_dir_path),
            desc="Counting all categorical masks irregular pixels...",
            colour="yellow",
        )
    }
    save_dict_to_csv(dict_to_save, output_path)


def get_image_with_more_than_irregular_pixels_limit(
    irregular_pixels_limit: int,
    all_categorical_mask_irregular_pixels_count_save_path: Path,
):
    saved_dict = load_saved_dict(all_categorical_mask_irregular_pixels_count_save_path)
    irregular_pixel_count_dict = {
        image_path: sum(ast.literal_eval(count_dict).values())
        for image_path, count_dict in saved_dict.items()
    }
    filtered_with_count_limit_dict = {
        image_path: count
        for image_path, count in irregular_pixel_count_dict.items()
        if count > irregular_pixels_limit
    }
    return filtered_with_count_limit_dict
