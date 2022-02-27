import uuid
import imageio
import tensorflow as tf
from pathlib import Path
from loguru import logger

from utils.image_utils import decode_image


# Variables
MASKS_DIR = Path(
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/marie"
)


def merge_all_masks(masks_dir: Path) -> None:
    """

    :param masks_dir: Path of the folder where the images masks folder are.
    """
    for image_masks_sub_dir in masks_dir.iterdir():
        for class_sub_dir in image_masks_sub_dir.iterdir():
            if len(list(class_sub_dir.iterdir())) > 1:
                masks_to_merge_paths_list = list(class_sub_dir.iterdir())
                image_name = image_masks_sub_dir.parts[-1]
                class_name = image_masks_sub_dir.parts[-1]
                output_file_name = class_sub_dir / (
                    "merged_mask_"
                    + image_name
                    + "_"
                    + class_name
                    + "__"
                    + uuid.uuid4().hex
                    + ".png"
                )
                create_merged_mask(masks_to_merge_paths_list, output_file_name)


def create_merged_mask(
    masks_to_merge_paths_list: [Path], output_file_name: Path
) -> None:
    """
    Create a merged mask for images with more than 1 mask for the same class.

    :param masks_to_merge_paths_list:
    :param output_file_name:
    """
    first_mask = decode_image(masks_to_merge_paths_list[0])
    for class_mask_path in masks_to_merge_paths_list[1:]:
        first_mask = tf.add(first_mask, decode_image(class_mask_path))
    imageio.imwrite(output_file_name, first_mask)
    logger.info(f"\nMerged mask {output_file_name} created successfully.")


def get_masks_to_merge(masks_dir: Path) -> [Path]:
    """

    :param masks_dir: Path of the folder where the images masks folder are.
    :return: List of list of masks to merge together.
    """
    masks_to_merge = list()
    for image_masks_subdir in masks_dir.iterdir():
        for class_subdir in image_masks_subdir.iterdir():
            if len(list(class_subdir.iterdir())) > 1:
                masks_to_merge.append(list(class_subdir.iterdir()))
    return masks_to_merge
