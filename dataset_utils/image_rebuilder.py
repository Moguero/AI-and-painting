from pathlib import Path
from loguru import logger

import tensorflow as tf

from dataset_utils.image_utils import decode_image

IMAGE_PATCHES_DIR = Path(
    r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\patches\1"
)
ORIGINAL_IMAGE_PATH = Path(
    r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\images\1\1.jpg"
)
PATCH_SIZE = 256


def rebuild_image(image_patches_dir: Path, original_image_path: Path, patch_size: int) -> tf.Tensor:
    original_image_tensor = decode_image(original_image_path)
    n_horizontal_patches = original_image_tensor.shape[0] // patch_size
    n_vertical_patches = original_image_tensor.shape[1] // patch_size

    assert n_horizontal_patches * n_vertical_patches == len(
        list(image_patches_dir.iterdir())
    ), f"The number of patches is not the same : original image should have {n_vertical_patches*n_horizontal_patches} while we have {len(list(image_patches_dir.iterdir()))} "

    for row_number in range(n_horizontal_patches):
        for column_number in range(n_vertical_patches):
            patch_number = row_number * n_vertical_patches + column_number + 1
            for patch_path in (image_patches_dir / str(patch_number) / "image").iterdir(): # loop of size 1
                patch_tensor = decode_image(patch_path)
                if column_number == 0:
                    line_rebuilt_tensor = patch_tensor
                else:
                    line_rebuilt_tensor = tf.concat([line_rebuilt_tensor, patch_tensor], axis=1)
        if row_number == 0:
            rebuilt_tensor = line_rebuilt_tensor
        else:
            rebuilt_tensor = tf.concat([rebuilt_tensor, line_rebuilt_tensor], axis=0)
    logger.info(f"\nImage of original size {original_image_tensor.shape} has been rebuilt with size {rebuilt_tensor.shape}")
    return rebuilt_tensor


# rebuild_image(IMAGE_PATCHES_DIR, ORIGINAL_IMAGE_PATH, PATCH_SIZE)
