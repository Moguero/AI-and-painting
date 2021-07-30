from pathlib import Path
import tensorflow as tf
from loguru import logger

from dataset_utils.image_utils import (
    decode_image,
    get_image_name_without_extension,
    get_images_paths,
)
from dataset_utils.mask_processing import save_tensor_to_jpg

IMAGES_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images")
IMAGES_PATCHES_DIR_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images_patches")
IMAGE_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/1/1.jpg")
OUTPUT_DIR_PATH = Path(
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images_patches/"
)
PATCH_SIZE = 256


# todo : add padding to the image if its shape is not a multiple of the patch_size
# todo : save this padding in the labels_masking not to train on it
# todo : add the the labels in this function to process them as well
def extract_image_patches(
    image_path: Path,
    patch_size: int,
    padding: str = "VALID",
) -> [tf.Tensor]:
    sizes = [1, patch_size, patch_size, 1]
    strides = [1, patch_size, patch_size, 1]
    rates = [1, 1, 1, 1]
    image = decode_image(image_path)
    image = tf.expand_dims(image, 0)

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
        output_path = output_subdir_path / f"patch_{idx + 1}.jpg"
        save_tensor_to_jpg(patch_tensor, output_path)


def save_all_images_patches(
    images_dir: Path,
    images_patches_dir_path: Path,
    patch_size: int,
    padding: str = "VALID",
):
    image_dir_paths = get_images_paths(images_dir)
    for image_path in image_dir_paths:
        image_name = get_image_name_without_extension(image_path)
        output_sub_dir = images_patches_dir_path / image_name
        save_image_patches(image_path, patch_size, output_sub_dir, padding)


# extract_image_patches(IMAGE_PATH, PATCH_SIZE, OUTPUT_DIR_PATH)
# save_image_patches(IMAGE_PATH, PATCH_SIZE, OUTPUT_DIR_PATH)
