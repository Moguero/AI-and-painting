from loguru import logger
from tqdm import tqdm

from constants import MAPPING_CLASS_NUMBER, MASK_TRUE_VALUE, MASK_FALSE_VALUE
from dataset_utils.file_utils import save_list_to_csv, load_saved_list
from dataset_utils.image_utils import (
    get_image_masks_paths,
    get_mask_class,
    get_image_name_without_extension,
    get_images_paths,
    get_image_patch_masks_paths,
)
from pathlib import Path
import tensorflow as tf
from dataset_utils.image_utils import decode_image


def stack_image_patch_masks(
    image_patch_masks_paths: [Path],
    mapping_class_number: {str: int},
) -> tf.Tensor:
    """

    :param image_patch_masks_shape: A tuple (width, height) only : no channels dim !
    """
    assert "background" in list(
        mapping_class_number.keys()
    ), f"'background' class is not specified in the class/number mapping : {mapping_class_number}"
    shape = decode_image(file_path=image_patch_masks_paths[0])[:, :, 0].shape
    stacked_tensor = tf.zeros(shape=shape, dtype=tf.int32)

    problematic_indices_list = list()
    zeros_tensor = tf.zeros(shape=shape, dtype=tf.int32)
    for patch_mask_path in image_patch_masks_paths:
        class_categorical_tensor = turn_mask_into_categorical_tensor(
            mask_path=patch_mask_path
        )

        # spotting the problematic pixels indices
        background_mask_class_categorical_tensor = tf.logical_not(
            tf.equal(class_categorical_tensor, zeros_tensor)
        )
        background_mask_stacked_tensor = tf.logical_not(
            tf.equal(stacked_tensor, zeros_tensor)
        )
        logical_tensor = tf.logical_and(
            background_mask_class_categorical_tensor, background_mask_stacked_tensor
        )
        non_overlapping_tensor = tf.equal(
            logical_tensor, tf.constant(False, shape=shape)
        )
        problematic_indices = (
            tf.where(tf.equal(non_overlapping_tensor, False)).numpy().tolist()
        )
        if problematic_indices:
            problematic_indices_list += problematic_indices

        # adding the class mask to the all-classes-stacked tensor
        stacked_tensor = tf.math.add(stacked_tensor, class_categorical_tensor)

    # setting the irregular pixels to background class
    if problematic_indices_list:
        categorical_array = stacked_tensor.numpy()
        for pixels_coordinates in problematic_indices_list:
            categorical_array[tuple(pixels_coordinates)] = mapping_class_number[
                "background"
            ]
        stacked_tensor = tf.constant(categorical_array, dtype=tf.int32)

    # check that the problematic pixels were set to 0 correctly
    for pixels_coordinates in problematic_indices_list:
        assert (
            stacked_tensor[tuple(pixels_coordinates)].numpy()
            == mapping_class_number["background"]
        )

    return stacked_tensor


def one_hot_encode_image_patch_masks(
    image_patch_masks_paths: [Path],
    n_classes: int,
    mapping_class_number: {str: int},
) -> tf.Tensor:
    categorical_mask_tensor = stack_image_patch_masks(
        image_patch_masks_paths=image_patch_masks_paths,
        mapping_class_number=mapping_class_number,
    )
    one_hot_encoded_tensor = tf.one_hot(
        indices=categorical_mask_tensor, depth=n_classes + 1, dtype=tf.int32
    )
    return one_hot_encoded_tensor


def turn_mask_into_categorical_tensor(mask_path: Path) -> tf.Tensor:
    tensor_first_channel = decode_image(mask_path)[:, :, 0]
    mask_class = get_mask_class(mask_path)
    categorical_number = MAPPING_CLASS_NUMBER[mask_class]
    categorical_tensor = tf.where(
        tf.equal(tensor_first_channel, MASK_TRUE_VALUE),
        categorical_number,
        MASK_FALSE_VALUE,
    )
    return categorical_tensor


# todo : test write_file and read_file
def save_tensor_to_jpg(tensor: tf.Tensor, output_filepath: Path) -> None:
    file_name = output_filepath.parts[-1]
    assert (
        file_name[-3:] == "jpg"
        or file_name[-3:] == "JPG"
        or file_name[-3:] == "jpeg"
        or file_name[-3:] == "JPEG"
    ), f"The output path {output_filepath} is not with jpg format."
    encoded_image_tensor = tf.io.encode_jpeg(tensor)
    tf.io.write_file(
        filename=tf.constant(str(output_filepath)), contents=encoded_image_tensor
    )


def stack_image_masks(
    image_path: Path,
    masks_dir_path: Path,
) -> tf.Tensor:
    """
    Returns a stacked tensor, with a size corresponding to the number of pixels of image_path.

    :param image_path: The source image on which to compute the stacked labels mask.
    :param masks_dir_path: The masks source directory path.
    :return: A 2D stacked tensor.
    """
    image_masks_paths = get_image_masks_paths(
        image_path=image_path, masks_dir_path=masks_dir_path
    )
    shape = decode_image(image_masks_paths[0])[:, :, 0].shape
    stacked_tensor = tf.zeros(shape=shape, dtype=tf.int32)

    problematic_indices_list = list()
    zeros_tensor = tf.zeros(shape=shape, dtype=tf.int32)
    for mask_path in image_masks_paths:
        class_categorical_tensor = turn_mask_into_categorical_tensor(
            mask_path=mask_path
        )

        # spotting the problematic pixels indices
        background_mask_class_categorical_tensor = tf.logical_not(
            tf.equal(class_categorical_tensor, zeros_tensor)
        )
        background_mask_stacked_tensor = tf.logical_not(
            tf.equal(stacked_tensor, zeros_tensor)
        )
        logical_tensor = tf.logical_and(
            background_mask_class_categorical_tensor, background_mask_stacked_tensor
        )
        non_overlapping_tensor = tf.equal(
            logical_tensor, tf.constant(False, shape=shape)
        )
        problematic_indices = (
            tf.where(tf.equal(non_overlapping_tensor, False)).numpy().tolist()
        )
        if problematic_indices:
            problematic_indices_list += problematic_indices

        # adding the class mask to the all-classes-stacked tensor
        stacked_tensor = tf.math.add(stacked_tensor, class_categorical_tensor)

    # setting the irregular pixels to background class
    if problematic_indices_list:
        categorical_array = stacked_tensor.numpy()
        for pixels_coordinates in problematic_indices_list:
            categorical_array[tuple(pixels_coordinates)] = MAPPING_CLASS_NUMBER[
                "background"
            ]
        stacked_tensor = tf.constant(categorical_array, dtype=tf.int32)

    # check that the problematic pixels were set to 0 correctly
    for pixels_coordinates in problematic_indices_list:
        assert (
            stacked_tensor[tuple(pixels_coordinates)].numpy()
            == MAPPING_CLASS_NUMBER["background"]
        )

    return stacked_tensor


# debug
def f():
    zeros_tensor = tf.zeros(shape=(3, 3), dtype=tf.int32)
    a = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mask_a = tf.logical_not(tf.equal(a, zeros_tensor))
    b = tf.constant([[0, 0, 0], [2, 2, 0], [0, 0, 0]])
    mask_b = tf.logical_not(tf.equal(b, zeros_tensor))
    logical_tensor = tf.logical_and(mask_a, mask_b)
    non_overlapping_tensor = tf.equal(logical_tensor, tf.constant(False, shape=(3, 3)))
    problematic_indices = (
        tf.where(tf.equal(non_overlapping_tensor, False)).numpy().tolist()
    )
    assert not problematic_indices, (
        f"\nMasks are overlapping."
        f"\nNumber of problematic pixels : {len(problematic_indices)}"
        f"\nProblematic pixels indices : {problematic_indices}"
    )
