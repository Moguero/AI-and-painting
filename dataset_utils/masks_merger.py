import uuid
from pathlib import Path

import imageio

from dataset_utils.mask_utils import get_image_name_without_extension, load_mask


# todo : optimize the mask merging with tensorflow
def merge_masks(image_path: Path, masks_dir_path: Path):
    """Merges masks of all the classes containing more than one mask."""
    image_masks_sub_dir = masks_dir_path / get_image_name_without_extension(image_path)
    for class_sub_dir in image_masks_sub_dir.iterdir():
        class_sub_dir_path = image_masks_sub_dir / "/" / class_sub_dir / "/"
        if len(list(class_sub_dir_path.iterdir())) > 1:
            masks_to_merge_paths_list = [
                class_sub_dir_path / class_mask_name
                for class_mask_name in class_sub_dir_path.iterdir()
            ]
            output_file_name = (
                class_sub_dir_path
                / "merged_mask_"
                / get_image_name_without_extension(image_path)
                / "_"
                / class_sub_dir
                / "__"
                / uuid.uuid4().hex
                / ".png"
            )
            create_merged_mask(masks_to_merge_paths_list, output_file_name)


def create_merged_mask(masks_to_merge_paths_list: list, output_file_name: Path):
    first_mask = load_mask(masks_to_merge_paths_list[0])
    for class_masks_path_idx in range(1, len(masks_to_merge_paths_list)):
        next_mask = load_mask(masks_to_merge_paths_list[class_masks_path_idx])
        first_mask = first_mask + next_mask
    imageio.imwrite(output_file_name, first_mask)
