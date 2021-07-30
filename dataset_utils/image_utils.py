import shutil
import time

from loguru import logger
from pathlib import Path
import tensorflow as tf

# from tests.test_labels import test_same_folders_of_images_and_masks, test_same_folders_of_images_and_categorical_masks

IMAGES_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images")
MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks")
CATEGORICAL_MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/categorical_masks")
DATASET_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/dataset")
FILES_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043")
IMAGE_SOURCE_PATH = Path(
    "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/sorted_images/unkept/Végétation 2/_DSC0069 - Copie.JPG"
)
IMAGE_TARGET_DIR_PATH = Path(
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/"
)
IMAGE_PATH = Path(
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043/_DSC0043.JPG"
)

IMAGE_SIZE = (160, 160)
N_CLASSES = 3
BATCH_SIZE = 32

IMAGES_SOURCE_DIR = Path("C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/sorted_images/kept/Adrien_images")

# todo : documenter toutes les fonctions


def get_image_name_without_extension(image_path: Path) -> str:
    return image_path.parts[-1].split(".")[0]


def get_file_name_with_extension(file_path: Path) -> str:
    return file_path.parts[-1]


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
    file_name = image_name + ".jpg"
    output_path = sub_dir / file_name
    if not Path(sub_dir).exists():
        Path(sub_dir).mkdir()
        logger.info(f"\nSub folder {sub_dir} was created")
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


def copy_images_with_masks(images_source_dir: Path, masks_dir: Path, image_target_dir_path: Path):
    names_of_images_with_masks = get_names_of_images_with_masks(masks_dir)
    for image_path in list(images_source_dir.iterdir()):
        image_name_without_extension = get_image_name_without_extension(image_path)
        if image_name_without_extension in names_of_images_with_masks:
            copy_image(image_path, image_target_dir_path)


def get_names_of_images_with_masks(masks_dir: Path) -> [Path]:
    image_names = [mask_image_dir.parts[-1] for mask_image_dir in list(masks_dir.iterdir())]
    return image_names


def get_dir_paths(dir_path: Path) -> [Path]:
    return list(dir_path.iterdir())


def get_images_paths(images_dir: Path) -> [Path]:
    return [image_path for image_dir_path in images_dir.iterdir() for image_path in image_dir_path.iterdir()]


def timeit(method):
    """Decorator to time the execution of a function."""

    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        print(f"{method.__name__} : {int(end_time - start_time)}s to execute")
        return result

    return timed


def group_images_and_all_masks_together(dataset_dir: Path, images_dir: Path, masks_dir: Path, categorical_masks_dir: Path) -> None:
    # Initital checks
    # test_same_folders_of_images_and_masks(images_dir, masks_dir)
    # test_same_folders_of_images_and_categorical_masks(images_dir, categorical_masks_dir)

    images_dirs_paths = get_dir_paths(images_dir)
    for image_dir_path in images_dirs_paths:
        image_dir_name = image_dir_path.parts[-1]
        dataset_subdir = dataset_dir / image_dir_name
        if not dataset_subdir.exists():
            dataset_subdir.mkdir()
            logger.info(f"\nSub folder {dataset_subdir} was created.")

        # Copy all the images to dataset_dir
        for image_path in image_dir_path.iterdir():   # loop of size 1
            image_name_with_extension = get_file_name_with_extension(image_path)
            output_path = dataset_subdir / image_name_with_extension
            shutil.copyfile(str(image_path), str(output_path))

        # Copy all the categorical masks to dataset_dir
        categorical_mask_dir_path = categorical_masks_dir / image_dir_name
        for categorical_mask_path in categorical_mask_dir_path.iterdir():  # loop of size 1
            output_path = dataset_subdir / get_file_name_with_extension(categorical_mask_path)
            shutil.copyfile(str(categorical_mask_path), str(output_path))

        # Copy all the binary masks to dataset_dir
        masks_dir_path = masks_dir / image_dir_name
        for class_dir_path in masks_dir_path.iterdir():  # loop of size 9 maximum
            for mask_path in class_dir_path.iterdir():
                output_path = dataset_subdir / get_file_name_with_extension(mask_path)
                shutil.copyfile(str(mask_path), str(output_path))
