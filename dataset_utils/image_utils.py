import shutil
from loguru import logger
from pathlib import Path
from dataset_utils.mask_utils import get_image_name_without_extension

IMAGES_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images")
MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks")
FILES_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043")
IMAGE_SOURCE_PATH = Path(
    "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/sorted_images/unkept/Végétation 2/_DSC0069 - Copie.JPG"
)
IMAGE_TARGET_DIR_PATH = Path(
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/test/"
)

IMAGE_SIZE = (160, 160)
N_CLASSES = 3
BATCH_SIZE = 32


# todo : merge image_utils and mask_utils

def copy_image(image_source_path: Path, image_target_dir_path: Path):
    """Copy an image from a source path to a target path"""
    image_name = get_image_name_without_extension(image_source_path)
    sub_dir = image_target_dir_path / image_name
    file_name = image_name + ".png"
    output_path = sub_dir / file_name
    if not Path(sub_dir).exists():
        Path(sub_dir).mkdir()
        logger.info(f"Sub folder {sub_dir} was created")
    shutil.copyfile(str(image_source_path), str(output_path))


def get_files_paths(files_dir: Path):
    """Get the paths of the files (not folders) in a given folder."""
    file_paths = sorted([files_dir / file_name for file_name in files_dir.iterdir() if file_name.is_file()])
    return file_paths


# Some logs

image_paths = get_files_paths(IMAGES_DIR)
mask_paths = get_files_paths(MASKS_DIR)

print("Number of images:", len(get_files_paths(IMAGES_DIR)))

for input_path, target_path in zip(image_paths, mask_paths):
    print(input_path, "|", target_path)
