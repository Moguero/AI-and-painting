from pathlib import Path
from loguru import logger

import tensorflow as tf

from constants import PATCHES_DIR_PATH, IMAGE_PATH, PATCH_SIZE, PATCH_OVERLAP
from dataset_utils.image_cropping import crop_patch_tensor
from dataset_utils.image_utils import decode_image, get_tensor_dims, get_image_tensor_shape


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
    main_patch_classes_list: [tf.Tensor],
    right_side_patch_classes_list: [tf.Tensor],
    image_tensor: tf.Tensor,
    patch_size: int,
    patch_overlap: int,
    misclassification_size: int = 5,
) -> tf.Tensor:
    """
    Restructure the patches that were generated with the extract_patches_with_overlap() function.
    Warning : This function is strongly coupled with the function extract_patches() from the patches_generator.py module

    :param main_patch_classes_list: List with size n_patches of tensor with size (patch_size, patch_size)
    :param right_side_patch_classes_list:
    :param image_tensor:
    :param patch_size: Size of the patches.
    :param patch_overlap: Number of pixels on which neighbors patches intersect each other.
    :param misclassification_size: Estimated number of pixels on which the classification is wrong due to side effects between neighbors patches.
    :return: The rebuilt image tensor with dimension : [width, height, 1].
    """
    assert (
        patch_overlap % 2 == 0
    ), f"Patch overlap argument must be a pair number. The one specified was {patch_overlap}."
    assert (
        misclassification_size <= patch_overlap / 2
    ), f"Please increase the patch overlap (currently {patch_overlap}) to at least 2 times the misclassification size (currently {misclassification_size})."

    # Turns the patches in the list to a size of (patch_size, patch_size, 1)
    main_patch_classes_list = [
        tf.expand_dims(input=patch_classes, axis=2)
        for patch_classes in main_patch_classes_list
    ]

    right_side_patch_classes_list = [
        tf.expand_dims(input=patch_classes, axis=2)
        for patch_classes in right_side_patch_classes_list
    ]

    # Counting the number of main patches by which the image has been cut
    image_height, image_width, channels_number = get_image_tensor_shape(
        image_tensor=image_tensor
    )

    window_stride = (
        patch_size - patch_overlap
    )  # number of pixels by which we shift the window at each step of predictions
    n_vertical_patches = image_height // window_stride
    n_horizontal_patches = image_width // window_stride

    logger.info("\nRebuilding predictions patches...")
    for row_number in range(n_vertical_patches):
        for column_number in range(n_horizontal_patches):
            # Rebuild the line
            patch_number = row_number * n_horizontal_patches + column_number
            patch_tensor = main_patch_classes_list[patch_number]

            # cropping the patch by taking into account the overlap with which it was built
            cropped_patch_tensor = crop_patch_tensor(
                patch_tensor=patch_tensor, patch_overlap=patch_overlap
            )
            if column_number == 0:
                line_rebuilt_tensor = cropped_patch_tensor
            else:
                line_rebuilt_tensor = tf.concat(
                    [line_rebuilt_tensor, cropped_patch_tensor], axis=1
                )
        # Rebuild the row

        # first add the right side patches to extend the right side
        right_side_patch_tensor = right_side_patch_classes_list[row_number]
        cropped_right_side_patch_tensor = crop_patch_tensor(
            patch_tensor=right_side_patch_tensor, patch_overlap=patch_overlap
        )
        resized_cropped_right_side_patch_tensor = cropped_right_side_patch_tensor[
            :, image_width - (n_horizontal_patches * window_stride) :
        ]
        line_rebuilt_tensor = tf.concat(
            [line_rebuilt_tensor, resized_cropped_right_side_patch_tensor], axis=1
        )

        if row_number == 0:
            rebuilt_tensor = line_rebuilt_tensor
        else:
            rebuilt_tensor = tf.concat([rebuilt_tensor, line_rebuilt_tensor], axis=0)
    # todo : unhardcode the axis parameter in tf.concat

    # Check that the final size is consistent
    rebuilt_tensor_height, rebuilt_tensor_width, rebuilt_channels_number = get_image_tensor_shape(
        image_tensor=rebuilt_tensor
    )
    assert rebuilt_tensor_height == (
        image_height - 2 * (patch_overlap / 2)
    ), f"Number of rows is not consistent : got {rebuilt_tensor_height}, expected {image_height - 2 * (patch_overlap / 2)}"
    assert rebuilt_tensor_width == (
        image_width - 2 * (patch_overlap / 2)
    ), f"Number of columns is not consistent : got {rebuilt_tensor_width}, expected {image_width - 2 * (patch_overlap / 2)}"

    rebuilt_tensor = tf.squeeze(input=rebuilt_tensor)
    logger.info(
        f"\nImage predictions have been successfully built with size {rebuilt_tensor.shape} (original image size : {image_tensor.shape})."
    )
    return rebuilt_tensor


# -------
# DEBUG
# rebuild_image(PATCHES_DIR_PATH, IMAGE_PATH, PATCH_SIZE)
# a = rebuild_overlapping_patches_test(patches, IMAGE_PATH, PATCH_SIZE, PATCH_OVERLAP)
