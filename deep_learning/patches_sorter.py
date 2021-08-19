import tensorflow as tf
import numpy as np
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from dataset_utils.file_utils import save_list_to_csv, load_saved_dict, save_dict_to_csv, timeit
from dataset_utils.files_stats import count_mask_value_occurences_percent_of_2d_tensor
from dataset_utils.image_utils import get_file_name_with_extension, get_image_patch_paths
from dataset_utils.masks_encoder import stack_image_patch_masks


# DATA_DIR_ROOT = Path(r"/home/ec2-user/data")
DATA_DIR_ROOT = Path(r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files")
PATCHES_DIR_PATH = DATA_DIR_ROOT / "patches"
ONLY_BACKGROUND_PATCHES_PATH = DATA_DIR_ROOT / "temp_files/only_background_patches.csv"
PATCHES_COVERAGE_PATH = Path(
    r"C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/files/temp_files/patches_coverage.csv"
)
IMAGE_PATCH_PATH = DATA_DIR_ROOT / "patches/_DSC0038/95/image/patch_95.jpg"
SAVED_PATCHES_COVERAGE_PERCENT_PATH = DATA_DIR_ROOT / "temp_files/patches_coverage.csv"
ALL_PATCH_MASKS_OVERLAP_INDICES_PATH = DATA_DIR_ROOT / "temp_files/all_patch_masks_overlap_indices.csv"
SAVE_STATS_DIR_PATH = DATA_DIR_ROOT / "stats"
PATCHES_DIR_PATH = DATA_DIR_ROOT / "patches2"
PATCH_SIZE = 256


def get_patch_coverage(image_patch_path: Path) -> float:
    mask_tensor = stack_image_patch_masks(image_patch_path)
    count_mask_value_occurrence = count_mask_value_occurences_percent_of_2d_tensor(
        mask_tensor
    )
    if 0 not in count_mask_value_occurrence.keys():
        return 100
    else:
        return 100 - count_mask_value_occurrence[0]


def create_generator_patches_coverage(
    patches_dir_path: Path, all_masks_overlap_indices_path: Path
) -> [Path]:
    for image_dir_path in patches_dir_path.iterdir():
        for patch_dir_path in image_dir_path.iterdir():
            for image_patch_path in (patch_dir_path / "image").iterdir():
                yield image_patch_path, get_patch_coverage(image_patch_path, all_masks_overlap_indices_path)


def save_all_patches_coverage(patches_dir_path: Path, output_path: Path, all_masks_overlap_indices_path: Path) -> dict:
    patches_coverage_dict = dict()
    generator = create_generator_patches_coverage(patches_dir_path, all_masks_overlap_indices_path)
    while True:
        try:
            image_patch_path, patch_coverage = next(generator)
            patches_coverage_dict[image_patch_path] = patch_coverage
            logger.info(
                f"\nImage patch {image_patch_path} has a labels coverage of {patch_coverage}."
            )
        except StopIteration:
            break
    save_dict_to_csv(dict_to_export=patches_coverage_dict, output_path=output_path)
    return patches_coverage_dict


# @timeit
# def get_patches_above_coverage_percent_limit(
#     coverage_percent_limit: int, saved_patches_coverage_percent_path: Path
# ) -> [Path]:
#     patches_under_coverage_percent_limit_list = list()
#     patches_coverage_dict = load_saved_dict(saved_patches_coverage_percent_path)
#     for patch_path, coverage_percent in patches_coverage_dict.items():
#         if int(float(coverage_percent)) > coverage_percent_limit:
#             patches_under_coverage_percent_limit_list.append(patch_path)
#     return patches_under_coverage_percent_limit_list


@timeit
def get_patches_above_coverage_percent_limit(
    coverage_percent_limit: int,
    patches_dir: Path,
) -> [Path]:
    patches_under_coverage_percent_limit_list = list()
    image_patch_paths = get_image_patch_paths(patches_dir)
    for image_patch_path in tqdm(image_patch_paths, desc="Iterating over patches..."):
        coverage_percent = get_patch_coverage(image_patch_path)
        if int(float(coverage_percent)) > coverage_percent_limit:
            patches_under_coverage_percent_limit_list.append(image_patch_path)
    return patches_under_coverage_percent_limit_list


def is_patch_only_background(image_patch_path: Path, patch_size: int) -> bool:
    """
    Test if the labels of a patch is background only, i.e. a patch_size x patch_size array of zeros.
    """
    background_array = tf.zeros((patch_size, patch_size), dtype=tf.int32).numpy()
    patch_mask_array = stack_image_patch_masks(image_patch_path).numpy()
    try:
        np.testing.assert_array_equal(patch_mask_array, background_array)
        return True
    except AssertionError:
        return False


def get_only_background_patches_dir_paths(image_dir_path: Path, patch_size: int, all_masks_overlap_indices_path: Path) -> [Path]:
    only_background_patches_dir_paths = list()
    for image_patch_dir_path in image_dir_path.iterdir():
        for image_patch_path in (image_patch_dir_path / "image").iterdir():
            if is_patch_only_background(image_patch_path, patch_size, all_masks_overlap_indices_path):
                only_background_patches_dir_paths.append(image_patch_dir_path)
    return only_background_patches_dir_paths


def create_generator_all_only_background_patches(
    patches_dir_path: Path, patch_size: int, all_masks_overlap_indices_path: Path
) -> [Path]:
    for image_dir_path in patches_dir_path.iterdir():
        only_background_patches_dir_paths = get_only_background_patches_dir_paths(
            image_dir_path, patch_size, all_masks_overlap_indices_path
        )
        yield image_dir_path, only_background_patches_dir_paths


def save_all_only_background_patches_dir_paths(
    patches_dir_path: Path, patch_size: int, output_path: Path, all_masks_overlap_indices_path: Path
) -> [Path]:
    all_only_background_patches_dir_paths = list()
    generator = create_generator_all_only_background_patches(
        patches_dir_path, patch_size, all_masks_overlap_indices_path
    )
    while True:
        try:
            image_dir_path, only_background_patches_dir_paths = next(generator)
            all_only_background_patches_dir_paths += (
                only_background_patches_dir_paths
            )
            logger.info(
                f"\nImage {get_file_name_with_extension(image_dir_path)} has {len(only_background_patches_dir_paths)} patches with background only."
            )
        except StopIteration:
            break
    save_list_to_csv(
        list_to_export=all_only_background_patches_dir_paths,
        output_path=output_path,
    )
    return all_only_background_patches_dir_paths


def get_all_only_background_patches_dir_paths(
    saved_only_background_patches_dir_paths_path: Path
) -> [Path]:
    all_only_background_patches_dir_paths = list()
    only_background_patches_dir_paths_list = load_saved_dict(saved_only_background_patches_dir_paths_path)
    for only_background_patches_dir_path in only_background_patches_dir_paths_list:
        all_only_background_patches_dir_paths.append(only_background_patches_dir_path)
    return all_only_background_patches_dir_paths
