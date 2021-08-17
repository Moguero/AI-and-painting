import ast
from collections import Counter

import numpy as np
import tensorflow as tf
import os

from tqdm import tqdm

from constants import MASK_FALSE_VALUE, MASK_TRUE_VALUE
from dataset_utils.file_utils import save_dict_to_csv, load_saved_dict
from dataset_utils.image_utils import (
    decode_image,
    get_images_paths,
    get_file_name_with_extension,
    get_image_masks_paths, get_image_patch_masks_paths,
)
from pathlib import Path
from dataset_utils.masks_encoder import stack_image_masks


# DATA_DIR_ROOT = Path(r"/home/ec2-user/data")
DATA_DIR_ROOT = Path(
    "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/files"
)

IMAGE_PATH = DATA_DIR_ROOT / "images/1/1.png"
# MASKS_DIR = DATA_DIR_ROOT / "labels_masks"
MASKS_DIR = DATA_DIR_ROOT / "labels_masks/all"
IMAGES_DIR_PATH = DATA_DIR_ROOT / "images"
PATCHES_DIR = DATA_DIR_ROOT / "patches"
COUNT_ALL_CATEGORICAL_MASK_IRREGULAR_PIXELS_OUTPUT_PATH = (
    DATA_DIR_ROOT / "temp_files\categorical_mask_irregular_pixels_count.csv"
)
ALL_MASKS_OVERLAP_INDICES_OUTPUT_PATH = (
    DATA_DIR_ROOT / "temp_files/all_masks_overlap_indices.csv"
)
ALL_PATCH_MASKS_OVERLAP_INDICES_PATH = DATA_DIR_ROOT / "temp_files/all_patch_masks_overlap_indices.csv"
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
    image_path: Path, masks_dir: Path, all_patch_masks_overlap_indices_path: Path
) -> {int: float}:
    return count_mask_value_occurences_of_2d_tensor(
        stack_image_masks(image_path, masks_dir, all_patch_masks_overlap_indices_path)
    )


def count_categorical_mask_irregular_pixels(
    image_path: Path, masks_dir: Path, n_classes, all_patch_masks_overlap_indices_path: Path
) -> {int: int}:
    mask_value_occurences_of_categorical_mask_dict = (
        count_mask_value_occurences_of_categorical_mask(image_path, masks_dir, all_patch_masks_overlap_indices_path)
    )
    irregular_pixels_dict = {
        key: value
        for key, value in mask_value_occurences_of_categorical_mask_dict.items()
        if key > n_classes
    }
    return irregular_pixels_dict


def save_count_all_categorical_mask_irregular_pixels(
    images_dir_path: Path, masks_dir: Path, n_classes: int, output_path: Path, all_patch_masks_overlap_indices_path: Path
) -> None:
    dict_to_save = {
        image_path: count_categorical_mask_irregular_pixels(
            image_path, masks_dir, n_classes, all_patch_masks_overlap_indices_path
        )
        for image_path in tqdm(
            get_images_paths(images_dir_path),
            desc="Counting all categorical masks irregular pixels...",
            colour="yellow",
        )
    }
    breakpoint()
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


def save_all_masks_overlap_indices(
    images_dir_path: Path, masks_dir: Path, output_path
) -> None:
    masks_overlap_indices_dict = dict()
    for image_dir_path in tqdm(
        images_dir_path.iterdir(), desc="Getting overlap indices...", colour="yellow"
    ):
        for image_path in image_dir_path.iterdir(): # loop of size 1
            image_masks_paths = get_image_masks_paths(image_path, masks_dir)
            if len(image_masks_paths) != 1:
                shape = decode_image(image_masks_paths[0])[:, :, 0].shape
                stacked_tensor = tf.zeros(shape=shape, dtype=tf.int32)
                for mask_path in image_masks_paths:
                    stacked_tensor = tf.math.add(
                        stacked_tensor, tf.cast(decode_image(mask_path)[:, :, 0], tf.int32)
                    )
                stacked_values = list(
                    tf.unique_with_counts(tf.reshape(stacked_tensor, [-1])).y.numpy()
                )
                values_with_count_dict = count_mask_value_occurences_of_2d_tensor(
                    stacked_tensor
                )
                masks_overlap_indices_dict[image_path] = {
                    "n_overlap_indices": sum(
                        [
                            value
                            for key, value in values_with_count_dict.items()
                            if key != MASK_FALSE_VALUE and key != MASK_TRUE_VALUE
                        ]
                    ),
                    "problematic_indices": [
                        item
                        for indices_list in [
                            tf.where(tf.equal(stacked_tensor, value)).numpy().tolist()
                            for value in set(stacked_values)
                            - {MASK_FALSE_VALUE, MASK_TRUE_VALUE}
                        ]
                        for item in indices_list
                    ],
                }
    save_dict_to_csv(masks_overlap_indices_dict, output_path)


def save_all_patch_masks_overlap_indices(
    image_patches_dir_path: Path, output_path
) -> None:
    masks_overlap_indices_dict = dict()
    for image_dir_path in tqdm(
        image_patches_dir_path.iterdir(), desc="Getting overlap indices...", colour="yellow"
    ):
        for patch_dir_path in image_dir_path.iterdir():
            for patch_path in (patch_dir_path / "image").iterdir(): # loop of size 1
                image_masks_paths = get_image_patch_masks_paths(patch_path)
                if len(image_masks_paths) != 1:
                    shape = decode_image(image_masks_paths[0])[:, :, 0].shape
                    stacked_tensor = tf.zeros(shape=shape, dtype=tf.int32)
                    for mask_path in image_masks_paths:
                        stacked_tensor = tf.math.add(
                            stacked_tensor, tf.cast(decode_image(mask_path)[:, :, 0], tf.int32)
                        )
                    stacked_values = list(
                        tf.unique_with_counts(tf.reshape(stacked_tensor, [-1])).y.numpy()
                    )
                    values_with_count_dict = count_mask_value_occurences_of_2d_tensor(
                        stacked_tensor
                    )
                    masks_overlap_indices_dict[patch_path] = {
                        "n_overlap_indices": sum(
                            [
                                value
                                for key, value in values_with_count_dict.items()
                                if key != MASK_FALSE_VALUE and key != MASK_TRUE_VALUE
                            ]
                        ),
                        "problematic_indices": [
                            item
                            for indices_list in [
                                tf.where(tf.equal(stacked_tensor, value)).numpy().tolist()
                                for value in set(stacked_values)
                                - {MASK_FALSE_VALUE, MASK_TRUE_VALUE}
                            ]
                            for item in indices_list
                        ],
                    }
    breakpoint()
    save_dict_to_csv(masks_overlap_indices_dict, output_path)


def count_all_irregular_pixels(
    all_masks_overlap_indices_path: Path,
):
    all_masks_overlap_indices_dict = load_saved_dict(all_masks_overlap_indices_path)
    return sum(
        [
            value
            for image_path, masks_overlap_indices_dict in all_masks_overlap_indices_dict.items()
            for key, value in ast.literal_eval(masks_overlap_indices_dict).items()
            if key == "n_overlap_indices"
        ]
    )
