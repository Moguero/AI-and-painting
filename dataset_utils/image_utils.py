import shutil
import time

from loguru import logger
from pathlib import Path
import tensorflow as tf

IMAGES_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images")
MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks")
FILES_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043")
IMAGE_SOURCE_PATH = Path(
    "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/sorted_images/unkept/Végétation 2/_DSC0069 - Copie.JPG"
)
IMAGE_TARGET_DIR_PATH = Path(
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/test/"
)
IMAGE_PATH = Path(
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043/_DSC0043.JPG"
)

IMAGE_SIZE = (160, 160)
N_CLASSES = 3
BATCH_SIZE = 32

# todo : documenter toutes les fonctions


def get_image_name_without_extension(image_path: Path) -> str:
    return image_path.parts[-1].split(".")[0]


def get_image_name_with_extension(image_path: Path) -> str:
    return image_path.parts[-1]


def get_image_masks_paths(image_path: Path, masks_dir: Path) -> [Path]:
    """Get the paths of the image masks.

    Remark: If the masks sub directory associated to the image does not exist,
    an AssertionError will be thrown."""
    image_masks_sub_dir = masks_dir / get_image_name_without_extension(image_path)
    assert (
        image_masks_sub_dir.exists()
    ), f"Image masks sub directory {image_masks_sub_dir} does not exist"
    return [
        image_masks_sub_dir / class_mask_sub_dir / mask_name
        for class_mask_sub_dir in image_masks_sub_dir.iterdir()
        for mask_name in (image_masks_sub_dir / class_mask_sub_dir).iterdir()
    ]


def get_mask_class(mask_path: Path) -> str:
    return mask_path.parts[-2]


# todo : implement a try/except to decode only jpg or png (not based on image_type but metadata format of the image)
def decode_image(file_path: Path) -> tf.Tensor:
    """Turns a png or jpeg image into its tensor version."""
    value = tf.io.read_file(str(file_path))
    image_type = get_image_type(file_path)
    channels = get_image_channels_number(file_path)
    if image_type == "png" or image_type == "PNG":
        decoded_image = tf.image.decode_png(value, channels=channels)
    elif (
        image_type == "jpg"
        or image_type == "JPG"
        or image_type == "jpeg"
        or image_type == "JPEG"
    ):
        decoded_image = tf.image.decode_jpeg(value, channels=channels)
    else:
        decoded_image = tf.image.decode_image(value, channels=channels)
    return decoded_image


def get_image_type(image_path: Path) -> str:
    return image_path.parts[-1].split(".")[-1]


def get_image_channels_number(image_path: Path) -> int:
    image_type = get_image_type(image_path)
    image_channels_number = None
    if image_type == "png" or image_type == "PNG":
        image_channels_number = 4
    elif (
        image_type == "jpg"
        or image_type == "JPG"
        or image_type == "jpeg"
        or image_type == "JPEG"
    ):
        image_channels_number = 3
    assert (
        image_channels_number is not None
    ), f"Incorrect image type {image_type} of image with path {image_path}"
    return image_channels_number


def copy_image(image_source_path: Path, image_target_dir_path: Path) -> None:
    """Copy an image from a source path to a target path"""
    image_name = get_image_name_without_extension(image_source_path)
    sub_dir = image_target_dir_path / image_name
    file_name = image_name + ".png"
    output_path = sub_dir / file_name
    if not Path(sub_dir).exists():
        Path(sub_dir).mkdir()
        logger.info(f"Sub folder {sub_dir} was created")
    shutil.copyfile(str(image_source_path), str(output_path))


def get_files_paths(files_dir: Path) -> [Path]:
    """Get the paths of the files (not folders) in a given folder."""
    file_paths = sorted(
        [
            files_dir / file_name
            for file_name in files_dir.iterdir()
            if file_name.is_file()
        ]
    )
    return file_paths


# todo : type this as a function
def timeit(method):
    """Decorator to time the execution of a function."""

    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        print(f"{method.__name__} : {int(end_time - start_time)}s to execute")
        return result

    return timed
