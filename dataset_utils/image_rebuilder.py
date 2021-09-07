from pathlib import Path
from loguru import logger

import tensorflow as tf

from dataset_utils.image_cropping import crop_tensor
from dataset_utils.image_utils import decode_image, get_tensor_dims
from constants import *


def rebuild_image(
    image_patches_dir: Path, original_image_path: Path, patch_size: int
) -> tf.Tensor:
    original_image_tensor = decode_image(original_image_path)
    n_horizontal_patches = original_image_tensor.shape[0] // patch_size
    n_vertical_patches = original_image_tensor.shape[1] // patch_size

    assert n_horizontal_patches * n_vertical_patches == len(
        list(image_patches_dir.iterdir())
    ), f"The number of patches is not the same : original image should have {n_vertical_patches*n_horizontal_patches} while we have {len(list(image_patches_dir.iterdir()))} "

    for row_number in range(n_horizontal_patches):
        for column_number in range(n_vertical_patches):
            patch_number = row_number * n_vertical_patches + column_number + 1
            for patch_path in (
                image_patches_dir / str(patch_number) / "image"
            ).iterdir():  # loop of size 1
                patch_tensor = decode_image(patch_path)
                if column_number == 0:
                    line_rebuilt_tensor = patch_tensor
                else:
                    line_rebuilt_tensor = tf.concat(
                        [line_rebuilt_tensor, patch_tensor], axis=1
                    )
        if row_number == 0:
            rebuilt_tensor = line_rebuilt_tensor
        else:
            rebuilt_tensor = tf.concat([rebuilt_tensor, line_rebuilt_tensor], axis=0)
    logger.info(
        f"\nImage of original size {original_image_tensor.shape} has been rebuilt with size {rebuilt_tensor.shape}"
    )
    return rebuilt_tensor


# legacy function with side effects misclassifications for neighbors patches
def rebuild_predictions(
    predictions_patches: [tf.Tensor], target_image_path: Path, patch_size: int
) -> tf.Tensor:
    original_image_tensor = decode_image(target_image_path)
    n_horizontal_patches = original_image_tensor.shape[0] // patch_size
    n_vertical_patches = original_image_tensor.shape[1] // patch_size

    assert n_horizontal_patches * n_vertical_patches == len(
        predictions_patches
    ), f"The number of patches is not the same : original image should have {n_vertical_patches*n_horizontal_patches} while we have {len(predictions_patches)} "

    logger.info("\nRebuilding predictions patches...")
    for row_number in range(n_horizontal_patches):
        for column_number in range(n_vertical_patches):
            patch_number = row_number * n_vertical_patches + column_number
            prediction_patch = predictions_patches[patch_number]
            if column_number == 0:
                line_rebuilt_tensor = prediction_patch
            else:
                line_rebuilt_tensor = tf.concat(
                    [line_rebuilt_tensor, prediction_patch], axis=1
                )
        if row_number == 0:
            rebuilt_tensor = line_rebuilt_tensor
        else:
            rebuilt_tensor = tf.concat([rebuilt_tensor, line_rebuilt_tensor], axis=0)
    logger.info(
        f"\nFull image predictions has been successfully built with size {rebuilt_tensor.shape} (original image size : {original_image_tensor.shape})"
    )
    return rebuilt_tensor


# todo : set default value of misclassification_size correctly
def rebuild_predictions_with_overlap(
    patches: [tf.Tensor],
    target_image_path: Path,
    patch_size: int,
    patch_overlap: int,
    misclassification_size: int = 5,
) -> tf.Tensor:
    """
    Restructure the patches that were generated with the extract_patches_with_overlap() function.

    :param patches: List with size n_patches of tensor with size (patch_size, patch_size, 1)
    :param target_image_path: Path of the image we want to make predictions on.
    :param patch_size: Size of the patches.
    :param patch_overlap: Number of pixels on which neighbors patches intersect each other.
    :param misclassification_size: Estimated number of pixels on which the classification is wrong due to side effects between neighbors patches.
    :return: The rebuilt image tensor with dimension : [width, height, 1].
    """
    assert (
        patch_overlap % 2 == 0
    ), f"Patch overlap argument must be a pair number. The one specified was {patch_overlap}."
    assert misclassification_size <= patch_overlap / 2, (
        f"Please increase the patch overlap to at least 2 times the misclassification size."
        f"Current misclassification size : {misclassification_size}"
        f"Current patch overlap : {patch_overlap}"
    )

    # counting the number of patches in which the image has been cut
    original_image_tensor = decode_image(target_image_path)
    image_width_index, image_height_index, image_channels_index = get_tensor_dims(
        original_image_tensor
    )
    n_horizontal_patches = (original_image_tensor.shape[image_width_index] - patch_overlap) // (
        patch_size - patch_overlap
    )
    n_vertical_patches = (original_image_tensor.shape[image_height_index] - patch_overlap) // (
        patch_size - patch_overlap
    )
    assert n_horizontal_patches * n_vertical_patches == len(
        patches
    ), f"The number of patches is not the same : original image should have {n_vertical_patches*n_horizontal_patches} while we have {len(patches)} "

    logger.info("\nRebuilding predictions patches...")
    for row_number in range(n_horizontal_patches):
        for column_number in range(n_vertical_patches):
            patch_number = row_number * n_vertical_patches + column_number
            patch = patches[patch_number]

            # cropping the patch by taking into account the overlap with which it was built
            (
                patch_width_index,
                patch_height_index,
                patch_channels_index,
            ) = get_tensor_dims(patch)
            target_width = int(patch.shape[patch_width_index] - 2 * (patch_overlap / 2))
            target_height = int(
                patch.shape[patch_height_index] - 2 * (patch_overlap / 2)
            )
            patch = crop_tensor(
                patch, target_height=target_height, target_width=target_width
            )
            if column_number == 0:
                line_rebuilt_tensor = patch
            else:
                line_rebuilt_tensor = tf.concat([line_rebuilt_tensor, patch], axis=1)
        if row_number == 0:
            rebuilt_tensor = line_rebuilt_tensor
        else:
            rebuilt_tensor = tf.concat([rebuilt_tensor, line_rebuilt_tensor], axis=0)
    # todo : unhardcode the axis parameter in tf.concat
    rebuilt_width_index, rebuilt_height_index, rebuilt_channels_index = get_tensor_dims(
        rebuilt_tensor
    )

    # checking that the final size is consistent
    assert rebuilt_tensor.shape[rebuilt_width_index] == int(
        n_horizontal_patches * patch_size - (n_horizontal_patches - 1) * patch_overlap - 2 * (patch_overlap / 2)
    ), f"Number of rows is not consistent : got {rebuilt_tensor.shape[rebuilt_width_index]}, expected {int(n_horizontal_patches * patch_size - 2 * (patch_overlap / 2))}"
    assert rebuilt_tensor.shape[rebuilt_height_index] == int(
        n_vertical_patches * patch_size - (n_vertical_patches - 1) * patch_overlap - 2 * (patch_overlap / 2)
    ), f"Number of columns is not consistent : got {rebuilt_tensor.shape[rebuilt_height_index]}, expected {int(n_vertical_patches * patch_size - 2 * (patch_overlap / 2))}"

    logger.info(
        f"\nFull image predictions has been successfully built with size {rebuilt_tensor.shape}."
        f"\nOriginal image size : {original_image_tensor.shape}"
    )
    return rebuilt_tensor


# -------
# DEBUG
# rebuild_image(IMAGE_PATCHES_DIR, ORIGINAL_IMAGE_PATH, PATCH_SIZE)
# a = rebuild_overlapping_patches_test(patches, IMAGE_PATH, PATCH_SIZE, PATCH_OVERLAP)
