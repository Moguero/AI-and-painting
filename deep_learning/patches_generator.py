import time
from pathlib import Path
import tensorflow as tf
from loguru import logger

from dataset_utils.image_utils import (
    decode_image,
    get_image_name_without_extension,
    get_images_paths, get_image_masks_paths,
)
from dataset_utils.masks_encoder import save_tensor_to_jpg

IMAGES_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images")
IMAGES_PATCHES_DIR_PATH = Path(
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/patches"
)
IMAGE_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/1/1.jpg")
OUTPUT_DIR_PATH = Path(
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/patches/"
)
PATCH_SIZE = 256
CATEGORICAL_MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/categorical_masks")
MASKS_DIR = Path(r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\labels_masks\all")


# todo : add padding to the images if its shape is not a multiple of the patch_size
# todo : save this padding in the labels_masking not to train on it
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
    output_subdir_path = (
        output_dir_path / get_image_name_without_extension(image_path)
    )
    if not output_subdir_path.exists():
        output_subdir_path.mkdir()
        logger.info(f"\nSub folder {output_subdir_path} was created.")
    for idx, patch_tensor in enumerate(image_patches):
        image_path_sub_dir = output_subdir_path / str(idx + 1) / "image"
        if not output_subdir_path.exists():
            image_path_sub_dir.mkdir(parents=True)
        output_path = image_path_sub_dir / f"{get_image_name_without_extension(image_path)}_patch_{idx + 1}.jpg"
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
            image_path=image_mask_path, patch_size=patch_size, padding=padding, with_four_channels=True
        )
        output_subdir_path = output_dir_path / get_image_name_without_extension(image_path)
        if not output_subdir_path.exists():
            output_subdir_path.mkdir()
            logger.info(f"\nSub folder {output_subdir_path} was created.")
        for idx, patch_tensor in enumerate(image_labels_patches):
            class_name = image_mask_path.parts[-2]
            labels_patch_sub_dir = output_subdir_path / str(idx + 1) / ("labels/" + class_name)
            if not output_subdir_path.exists():
                labels_patch_sub_dir.mkdir(parents=True)
            output_path = labels_patch_sub_dir / f"{get_image_name_without_extension(image_path)}_patch_{idx + 1}_labels_{class_name}.jpg"
            save_tensor_to_jpg(patch_tensor, output_path)


def save_image_and_labels_patches(
    image_path: Path,
    masks_dir: Path,
    image_patches_dir_path: Path,
    patch_size: 256,
    padding: str = "VALID",
):
    save_image_patches(image_path, patch_size, image_patches_dir_path, padding)
    save_image_labels_patches(image_path, masks_dir, patch_size, image_patches_dir_path, padding)


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
    for image_path in image_dir_paths:
        save_image_and_labels_patches(image_path, masks_dir, images_patches_dir_path, patch_size, padding)
    logger.info(
        f"\nImages and labels patches saving finished in {(time.time() - start_time)/60:.1f} minutes.\n"
    )
