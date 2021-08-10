import tensorflow as tf
import numpy as np
from pathlib import Path
import shutil

from loguru import logger

from dataset_utils.file_utils import save_list_to_csv, load_saved_dict, save_dict_to_csv
from dataset_utils.files_stats import count_mask_value_occurences_percent_of_2d_tensor
from dataset_utils.image_utils import get_file_name_with_extension
from dataset_utils.masks_encoder import stack_image_patch_masks

IMAGE_PATCH_PATH = Path(
    r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\patches\1\1\image\patch_1.jpg"
)
IMAGE_DIR_PATH = Path(r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\patches\1")
PATCHES_DIR_PATH = Path(r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\patches")
SELECTED_PATCHES_DIR_PATH = Path(
    r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\selected_patches"
)
OUTPUT_PATH = Path(
    r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\temp_files\only_background_patches.csv"
)
SAVED_ONLY_BACKGROUND_PATCHES_PATH = OUTPUT_PATH
IMAGE_PATCH_PATH = Path(
    r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\patches\_DSC0038\95\image\patch_95.jpg"
)
PATCH_SIZE = 256


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


def get_patch_coverage(image_patch_path) -> float:
    mask_tensor = stack_image_patch_masks(image_patch_path)
    count_mask_value_occurence = count_mask_value_occurences_percent_of_2d_tensor(
        mask_tensor
    )
    if 0 not in count_mask_value_occurence.keys():
        return 100
    else:
        return 100 - count_mask_value_occurence[0]


# todo : rewrite this with a generator as it is done for background only functions
# todo : add a logger to this function
OUTPUT_PATH = Path(
    r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\temp_files\patches_coverage.csv"
)


def save_all_patches_coverage(patches_dir_path: Path, output_path: Path) -> None:
    dict_to_export = {
        image_patch_path: get_patch_coverage(image_patch_path)
        for image_dir_path in patches_dir_path.iterdir()
        for patch_dir_path in image_dir_path.iterdir()
        for image_patch_path in (patch_dir_path / "image").iterdir()  # loop of size 1
    }
    save_dict_to_csv(dict_to_export=dict_to_export, output_path=output_path)
    return dict_to_export


SAVED_PATCHES_COVERAGE_PERCENT_PATH = Path(r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\temp_files\patches_coverage.csv")
def get_patches_under_coverage_percent_limit(
    coverage_percent_limit: int, saved_patches_coverage_percent_path: Path
) -> [Path]:
    patches_under_coverage_percent_limit = list()
    patches_coverage_dict = load_saved_dict(saved_patches_coverage_percent_path)
    for patch_path, coverage_percent in patches_coverage_dict.items():
        if int(float(coverage_percent)) < coverage_percent_limit:
            patches_under_coverage_percent_limit.append(patch_path)
    return patches_under_coverage_percent_limit


# todo : rename into : get_only_background_patches_dir_paths
def get_only_background_patches(image_dir_path: Path, patch_size: int) -> [Path]:
    # todo : rename into : only_background_patches_dir_paths
    only_background_patches = list()
    for image_patch_dir_path in image_dir_path.iterdir():
        for image_patch_path in (image_patch_dir_path / "image").iterdir():
            if is_patch_only_background(image_patch_path, patch_size):
                only_background_patches.append(image_patch_dir_path)
    return only_background_patches


def create_generator_all_only_background_patches(
    patches_dir_path: Path, patch_size: int
) -> [Path]:
    for image_dir_path in patches_dir_path.iterdir():
        only_background_patches_dir_paths = get_only_background_patches(
            image_dir_path, patch_size
        )
        yield image_dir_path, only_background_patches_dir_paths


# todo : rename to get_all_only_background_patches_dir_paths
def get_all_only_background_patches(
    selected_patches_dir_path: Path, patch_size: int, output_path: Path, save_list=True
) -> [Path]:
    """
    Warning : this function must be use on the selected_patches folder, not the patches one.
    """
    all_only_background_patches_dir_paths = list()
    generator = create_generator_all_only_background_patches(
        selected_patches_dir_path, patch_size
    )
    while True:
        try:
            image_dir_path, only_background_patches_dir_paths = next(generator)
            if only_background_patches_dir_paths:
                all_only_background_patches_dir_paths += (
                    only_background_patches_dir_paths
                )
            logger.info(
                f"\nImage {get_file_name_with_extension(image_dir_path)} has {len(only_background_patches_dir_paths)} patches with background only."
            )
        except StopIteration:
            break
    if save_list:
        save_list_to_csv(
            list_to_export=all_only_background_patches_dir_paths,
            output_path=output_path,
        )
    return all_only_background_patches_dir_paths


def save_all_selected_patches(
    saved_only_background_patches_path: Path, selected_patches_dir_path: Path
):
    """
    Remark : we suppose that all the patches were already copied in a folder called selected_patches.
    This function will remove the non-selected patches from this folder.

    Warning : the saved_only_background_patches_path must have been created from the selected_patches folder, not the patches one.
    """
    only_background_patches_dir_paths = load_saved_dict(
        saved_only_background_patches_path
    )
    for patch_dir_path in only_background_patches_dir_paths:
        assert (
            patch_dir_path.parents[1] == selected_patches_dir_path
        ), f"Patch folder with path {patch_dir_path} is not in the selected_patches folder {selected_patches_dir_path}"
        shutil.rmtree(patch_dir_path)
        logger.info(f"\nDeleting patch folder {patch_dir_path}")
