import csv
import os
import ast
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from dataset_builder.masks_encoder import stack_image_masks, stack_image_patch_masks
from utils.image_utils import (
    decode_image,
    get_images_paths,
    get_file_name_with_extension,
    get_image_patch_masks_paths,
    get_image_patches_paths_with_limit,
)


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


def count_mask_value_occurences_percent_of_2d_tensor(tensor: tf.Tensor) -> {int: float}:
    unique_with_count_tensor = tf.unique_with_counts(tf.reshape(tensor, [-1]))
    values_array = unique_with_count_tensor.y.numpy()
    count_array = unique_with_count_tensor.count.numpy()
    percent_dict = dict(
        zip(values_array, np.round(count_array / count_array.sum() * 100, decimals=3))
    )
    return percent_dict


def get_patch_coverage(
    image_patch_masks_paths: [Path],
    mapping_class_number: {str: int},
) -> float:
    mask_tensor = stack_image_patch_masks(
        image_patch_masks_paths=image_patch_masks_paths,
        mapping_class_number=mapping_class_number,
    )
    count_mask_value_occurrence = count_mask_value_occurences_percent_of_2d_tensor(
        mask_tensor
    )
    if 0 not in count_mask_value_occurrence.keys():
        return 100
    else:
        return 100 - count_mask_value_occurrence[0]


def get_image_patches_paths(
    patches_dir_path: Path,
    batch_size: int,
    patch_coverage_percent_limit: int,
    test_proportion: float,
    mapping_class_number: {str: int},
    n_patches_limit: int = None,
    image_patches_paths: [Path] = None,
) -> [Path]:
    """
    Get images patches paths on which the model will train on.
    Also filter the valid patches above the coverage percent limit.

    :param patches_dir_path: Path of the patches root folder.
    :param n_patches_limit: Maximum number of patches used for the training.
    :param batch_size: Size of the batches.
    :param patch_coverage_percent_limit: Int, minimum coverage percent of a patch labels on this patch.
    :param test_proportion: Float, used to set the proportion of the test dataset.
    :param mapping_class_number: Mapping dictionary between class names and their representative number.
    :param image_patches_paths: If not None, list of patches to use to make the training on.
    :return: A list of paths of images to train on.
    """
    assert (
        0 <= test_proportion < 1
    ), f"Test proportion must be between 0 and 1 : {test_proportion} was given."
    logger.info("\nStart to build dataset...")
    if n_patches_limit is not None:
        assert (n_patches_limit // batch_size) * (
            1 - test_proportion
        ) >= 1, f"Size of training dataset is 0. Increase the n_patches_limit parameter or decrease the batch_size."

    logger.info("\nGet the paths of the valid patches for training...")
    patches_under_coverage_percent_limit_list = list()

    # if the image patches to train on are not provided, randomly select n_patches_limit patches
    if image_patches_paths is None:
        image_patches_paths = get_image_patches_paths_with_limit(
            patches_dir=patches_dir_path, n_patches_limit=n_patches_limit
        )

    for image_patch_path in tqdm(
        image_patches_paths,
        desc="Selecting patches above the coverage percent limit...",
    ):
        image_patch_masks_paths = get_image_patch_masks_paths(
            image_patch_path=image_patch_path
        )
        coverage_percent = get_patch_coverage(
            image_patch_masks_paths=image_patch_masks_paths,
            mapping_class_number=mapping_class_number,
        )
        if int(float(coverage_percent)) > patch_coverage_percent_limit:
            patches_under_coverage_percent_limit_list.append(image_patch_path)

    logger.info(
        f"\n{len(patches_under_coverage_percent_limit_list)}/{len(image_patches_paths)} patches above coverage percent limit selected."
    )
    return patches_under_coverage_percent_limit_list


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


def count_total_number_of_patches(patches_dir: Path) -> int:
    return sum([len(list(image_dir.iterdir())) for image_dir in patches_dir.iterdir()])


def count_mask_value_occurences_of_categorical_mask(
    image_path: Path, masks_dir: Path, all_patch_masks_overlap_indices_path: Path
) -> {int: float}:
    return count_mask_value_occurences_of_2d_tensor(
        stack_image_masks(image_path, masks_dir, all_patch_masks_overlap_indices_path)
    )


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


def get_patch_labels_composition(
    image_patch_masks_paths: [Path],
    n_classes: int,
    mapping_class_number: {str: int},
) -> {int: float}:
    """Compute the proportion (in %) of each class in the patch."""
    # initialize patch composition
    patch_composition = {class_number: 0.0 for class_number in range(n_classes + 1)}

    labels_tensor = stack_image_patch_masks(
        image_patch_masks_paths=image_patch_masks_paths,
        mapping_class_number=mapping_class_number,
    )
    unique_with_count_tensor = tf.unique_with_counts(tf.reshape(labels_tensor, [-1]))

    values_array = unique_with_count_tensor.y.numpy()
    count_array = unique_with_count_tensor.count.numpy()
    patch_class_proportions = dict(
        zip(
            values_array,
            count_array / (labels_tensor.shape[0] * labels_tensor.shape[1]),
        )
    )
    for class_number in patch_class_proportions.keys():
        patch_composition[class_number] = patch_class_proportions[class_number]

    assert sum(patch_composition.values()) == 1

    return patch_composition


def get_patches_labels_composition(
    image_patches_paths_list: [Path], n_classes: int, mapping_class_number: {str: int}
) -> pd.DataFrame:
    """
    For each patch of the list, returns the proportion of each class.
    """
    patch_composition_list = list()
    for image_patch_path in image_patches_paths_list:
        image_patch_masks_paths = get_image_patch_masks_paths(
            image_patch_path=image_patch_path
        )
        patch_composition = get_patch_labels_composition(
            image_patch_masks_paths=image_patch_masks_paths,
            n_classes=n_classes,
            mapping_class_number=mapping_class_number,
        )
        patch_composition_list.append(
            [proportion for proportion in patch_composition.values()]
        )
    patches_composition_dataframe = pd.DataFrame(
        patch_composition_list, columns=mapping_class_number.keys()
    )
    patches_composition_stats = patches_composition_dataframe.describe()

    assert (
        round(sum(list(patches_composition_stats.loc["mean"])), 2) == 1
    ), f'Total patches composition is {round(sum(list(patches_composition_stats.loc["mean"])), 2)}, should be 1.0'

    return patches_composition_stats


def save_list_to_csv(list_to_export: list, output_path: Path) -> None:
    assert (
        str(output_path)[-4:] == ".csv"
    ), f"Specified output path {output_path} is not in format .csv"
    with open(str(output_path), "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list_to_export)


def save_dict_to_csv(dict_to_export: dict, output_path: Path) -> None:
    assert (
        str(output_path)[-4:] == ".csv"
    ), f"Specified output path {output_path} is not in format .csv"
    with open(str(output_path), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for key, value in dict_to_export.items():
            writer.writerow([key, value])


def load_saved_list(input_path: Path) -> list:
    saved_list = list()
    with open(str(input_path), "r") as f:
        data = csv.reader(f)
        for row in data:
            saved_list += row
    return saved_list


def load_saved_dict(input_path: Path) -> dict:
    saved_dict = dict()
    with open(str(input_path), "r") as f:
        data = csv.reader(f)
        for row in data:
            saved_dict[row[0]] = row[1]
    return saved_dict


# -----
# DEBUG

# labels_patches_composition = get_patch_labels_composition(IMAGE_PATCH_PATH, N_CLASSES)
# patch_composition_dataframe = get_patches_labels_composition([IMAGE_PATCH_PATH], N_CLASSES, MAPPING_CLASS_NUMBER)
