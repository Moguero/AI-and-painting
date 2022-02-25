from pathlib import Path

from dataset_utils.file_utils import load_saved_dict
from dataset_utils.files_stats import count_mask_value_occurences_percent_of_2d_tensor
from dataset_utils.masks_encoder import stack_image_patch_masks


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


def get_all_only_background_patches_dir_paths(
    saved_only_background_patches_dir_paths_path: Path,
) -> [Path]:
    all_only_background_patches_dir_paths = list()
    only_background_patches_dir_paths_list = load_saved_dict(
        saved_only_background_patches_dir_paths_path
    )
    for only_background_patches_dir_path in only_background_patches_dir_paths_list:
        all_only_background_patches_dir_paths.append(only_background_patches_dir_path)
    return all_only_background_patches_dir_paths
