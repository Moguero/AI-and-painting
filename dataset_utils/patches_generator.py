import time
from pathlib import Path
import tensorflow as tf
from loguru import logger
from tqdm import tqdm

from dataset_utils.image_utils import (
    decode_image,
    get_image_name_without_extension,
    get_images_paths,
    get_image_masks_paths,
)
from dataset_utils.masks_encoder import save_tensor_to_jpg
from constants import *


def extract_image_patches(
    image_path: Path,
    patch_size: int,
    padding: str = "VALID",
    with_four_channels: bool = False,
) -> [tf.Tensor]:
    """
    Used for training images only : don't add padding to the borders.

    :param image_path: Path of the images that we want to patch.
    :param patch_size: Size of the patch. A regular value is 256.
    :param padding: Type of the padding to apply to the images. If "VALID", all the pixel of the patch are within the images.
    :param with_four_channels: If True, transforms the tensor image into a 3 channels tensor instead of 4.
    :return: Tensor of patches coming from the input images
    """
    sizes = [1, patch_size, patch_size, 1]
    strides = [1, patch_size, patch_size, 1]
    rates = [1, 1, 1, 1]
    image = decode_image(image_path)
    image = tf.expand_dims(image, 0)

    if with_four_channels:
        image = image[:, :, :, :3]

    patches_tensor = tf.image.extract_patches(
        image, sizes=sizes, strides=strides, rates=rates, padding=padding
    )
    reshaped_patches = tf.reshape(
        tensor=patches_tensor,
        shape=[
            patches_tensor.shape[1] * patches_tensor.shape[2],
            patch_size,
            patch_size,
            3,
        ],
    )
    splitted_reshaped_patches = tf.split(
        value=reshaped_patches, num_or_size_splits=reshaped_patches.shape[0], axis=0
    )
    squeezed_splitted_patches = [
        tf.squeeze(input=patch, axis=0) for patch in splitted_reshaped_patches
    ]
    return squeezed_splitted_patches


def save_image_patches(
    image_path: Path, patch_size, output_dir_path: Path, padding: str = "VALID"
) -> None:
    image_patches = extract_image_patches(
        image_path=image_path, patch_size=patch_size, padding=padding
    )
    output_subdir_path = output_dir_path / get_image_name_without_extension(image_path)
    if not output_subdir_path.exists():
        output_subdir_path.mkdir()
        logger.info(f"\nSub folder {output_subdir_path} was created.")
    for idx, patch_tensor in enumerate(image_patches):
        image_path_sub_dir = output_subdir_path / str(idx + 1) / "image"
        if not output_subdir_path.exists():
            image_path_sub_dir.mkdir(parents=True)
        output_path = (
            image_path_sub_dir
            / f"{get_image_name_without_extension(image_path)}_patch_{idx + 1}.jpg"
        )
        save_tensor_to_jpg(patch_tensor, output_path)


def save_image_labels_patches(
    image_path: Path,
    masks_dir: Path,
    patch_size,
    output_dir_path: Path,
    padding: str = "VALID",
) -> None:
    image_masks_paths = get_image_masks_paths(image_path, masks_dir)
    for image_mask_path in image_masks_paths:
        image_labels_patches = extract_image_patches(
            image_path=image_mask_path,
            patch_size=patch_size,
            padding=padding,
            with_four_channels=True,
        )
        output_subdir_path = output_dir_path / get_image_name_without_extension(
            image_path
        )
        if not output_subdir_path.exists():
            output_subdir_path.mkdir()
            logger.info(f"\nSub folder {output_subdir_path} was created.")
        for idx, patch_tensor in enumerate(image_labels_patches):
            class_name = image_mask_path.parts[-2]
            labels_patch_sub_dir = (
                output_subdir_path / str(idx + 1) / ("labels/" + class_name)
            )
            if not output_subdir_path.exists():
                labels_patch_sub_dir.mkdir(parents=True)
            output_path = (
                labels_patch_sub_dir
                / f"{get_image_name_without_extension(image_path)}_patch_{idx + 1}_labels_{class_name}.jpg"
            )
            save_tensor_to_jpg(patch_tensor, output_path)


def save_image_and_labels_patches(
    image_path: Path,
    masks_dir: Path,
    image_patches_dir_path: Path,
    patch_size: 256,
    padding: str = "VALID",
):
    save_image_patches(image_path, patch_size, image_patches_dir_path, padding)
    save_image_labels_patches(
        image_path, masks_dir, patch_size, image_patches_dir_path, padding
    )


def save_all_images_and_labels_patches(
    images_dir: Path,
    masks_dir: Path,
    images_patches_dir_path: Path,
    patch_size: int,
    padding: str = "VALID",
):
    logger.info("\nStarting to save images and labels patches...")
    start_time = time.time()
    image_dir_paths = get_images_paths(images_dir)
    for image_path in tqdm(image_dir_paths):
        save_image_and_labels_patches(
            image_path, masks_dir, images_patches_dir_path, patch_size, padding
        )
    logger.info(
        f"\nImages and labels patches saving finished in {(time.time() - start_time)/60:.1f} minutes.\n"
    )


def extract_patches_with_overlap(
    image_tensor: tf.Tensor,
    patch_size: int,
    patch_overlap: int,
    with_four_channels: bool = False,
) -> [tf.Tensor]:
    """
    Split an image into smaller patches.
    Padding is by default implemented as "VALID", meaning that only patches which are fully
    contained in the input image are included.

    :param image_tensor: Path of the image we want to cut into patches.
    :param patch_size: Size of the patch.
    :param patch_overlap: Number of pixels on which neighbors patches intersect each other.
    :param with_four_channels: Set it to True if the image is a PNG. Default to False for JPEG.
    :return: A list of patches of the original image.
    """
    image = tf.expand_dims(image_tensor, 0)

    if with_four_channels:
        image = image[:, :, :, :3]

    # todo : use get_image_shape
    n_rows = image.shape[1]
    n_columns = image.shape[2]

    patches = list()
    # patches_indices = list()

    row_idx = 0
    while row_idx + patch_size - 1 < n_rows:
        column_idx = 0
        while column_idx + patch_size - 1 < n_columns:
            patch = image[
                :,
                row_idx: row_idx + patch_size,
                column_idx: column_idx + patch_size,
                :,
            ]
            # dropping the first dimension
            patches.append(patch[0])
            # patch_indices = ((row_idx, row_idx + patch_size - 1), (column_idx, column_idx+ patch_size - 1))
            # patches_indices.append(patch_indices)
            column_idx += patch_size - patch_overlap
        row_idx += patch_size - patch_overlap
    return patches

# ------
# DEBUG
# save_all_images_and_labels_patches(IMAGES_DIR, MASKS_DIR, IMAGES_PATCHES_DIR_PATH, PATCH_SIZE)