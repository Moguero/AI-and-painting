from loguru import logger

from dataset_utils.image_utils import get_image_masks_paths, get_mask_class, get_image_name_without_extension, \
    get_images_paths, get_image_patch_masks_paths
from pathlib import Path
import tensorflow as tf
from dataset_utils.image_utils import decode_image
from constants import MAPPING_CLASS_NUMBER, MASK_TRUE_VALUE, MASK_FALSE_VALUE

IMAGE_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/1/1.jpg")
MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/all")
MASK_PATH = Path(r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\labels_masks\all\1\feuilles-vertes\mask_1_feuilles-vertes__090f44ab03ee43d7aaabe92aa58b06c1.png")
IMAGE_TYPE = "png"
OUTPUT_FILEPATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/test.jpg")
CHANNELS = 4
N_CLASSES = 9
CATEGORICAL_MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/categorical_masks")
IMAGES_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images")
CATEGORICAL_MASK_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/categorical_masks/_DSC0030/mask__DSC0030.jpg")
IMAGE_PATCH_PATH = Path(r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\patches\1\1\image\patch_1.jpg")


def get_mask_first_channel(mask_path: Path) -> tf.Tensor:
    decoded_image = decode_image(mask_path)
    return decoded_image[:, :, 0]


def stack_image_masks(image_path: Path, masks_dir: Path) -> tf.Tensor:
    """Fetches the images mask first channels, transform it into a categorical tensor and add the categorical together."""
    image_masks_paths = get_image_masks_paths(image_path, masks_dir)
    shape = get_mask_first_channel(image_masks_paths[0]).shape
    stacked_tensor = tf.zeros(shape=shape, dtype=tf.int32)
    for mask_path in image_masks_paths:
        categorical_tensor = turn_mask_into_categorical_tensor(mask_path)
        stacked_tensor = tf.math.add(stacked_tensor, categorical_tensor)
    return stacked_tensor


def stack_image_patch_masks(image_patch_path: Path) -> tf.Tensor:
    """Fetches the images mask first channels, transform it into a categorical tensor and add the categorical together."""
    image_masks_paths = get_image_patch_masks_paths(image_patch_path)
    shape = get_mask_first_channel(image_masks_paths[0]).shape
    stacked_tensor = tf.zeros(shape=shape, dtype=tf.int32)
    for mask_path in image_masks_paths:
        categorical_tensor = turn_mask_into_categorical_tensor(mask_path)
        stacked_tensor = tf.math.add(stacked_tensor, categorical_tensor)
    return stacked_tensor


def turn_mask_into_categorical_tensor(mask_path: Path) -> tf.Tensor:
    tensor_first_channel = get_mask_first_channel(mask_path)
    mask_class = get_mask_class(mask_path)
    categorical_number = MAPPING_CLASS_NUMBER[mask_class]
    categorical_tensor = tf.where(tf.equal(tensor_first_channel, MASK_TRUE_VALUE), categorical_number, MASK_FALSE_VALUE)
    return categorical_tensor


def one_hot_encode_image_masks(image_path: Path, categorical_masks_dir: Path, n_classes: int) -> tf.Tensor:
    categorical_mask_tensor = get_mask_first_channel(get_categorical_mask_path(image_path, categorical_masks_dir))
    one_hot_encoded_tensor = tf.one_hot(categorical_mask_tensor, n_classes, dtype=tf.int32)
    return one_hot_encoded_tensor


def one_hot_encode_image_patch_masks(image_patch_path: Path, n_classes: int) -> tf.Tensor:
    categorical_mask_tensor = stack_image_patch_masks(image_patch_path)
    one_hot_encoded_tensor = tf.one_hot(categorical_mask_tensor, n_classes + 1, dtype=tf.int32)
    return one_hot_encoded_tensor


# todo : test write_file and read_file
def save_tensor_to_jpg(tensor: tf.Tensor, output_filepath: Path) -> None:
    file_name = output_filepath.parts[-1]
    assert file_name[-3:] == "jpg" or file_name[-3:] == "JPG" or file_name[-3:] == "jpeg" or file_name[-3:] == "JPEG", f"The output path {output_filepath} is not with jpg format."
    encoded_image_tensor = tf.io.encode_jpeg(tensor)
    tf.io.write_file(filename=tf.constant(str(output_filepath)), contents=encoded_image_tensor)


def save_categorical_mask(image_path: Path, masks_dir: Path, output_filepath: Path) -> None:
    """Encode a categorical tensor into a png."""
    stacked_image_tensor = tf.expand_dims(tf.cast(stack_image_masks(image_path, masks_dir), tf.uint8), -1)
    save_tensor_to_jpg(stacked_image_tensor, output_filepath)
    logger.info(f"\nCategorical mask {output_filepath} was saved successfully.")


# todo : mkdir in the save_categorical_mask function instead of the save_all one
def save_all_categorical_masks(images_dir: Path, masks_dir: Path, categorical_masks_dir: Path) -> None:
    image_dir_paths = get_images_paths(images_dir)
    for image_path in image_dir_paths:
        image_name = get_image_name_without_extension(image_path)
        output_sub_dir = categorical_masks_dir / image_name
        if not output_sub_dir.exists():
            output_sub_dir.mkdir()
            logger.info(f"\nSub folder {output_sub_dir} was created.")
        output_path = output_sub_dir / ("categorical_mask__" + image_name + ".jpg")
        save_categorical_mask(image_path, masks_dir, output_path)


def get_categorical_mask_path(image_path: Path, categorical_masks_dir: Path):
    image_name = get_image_name_without_extension(image_path)
    categorical_masks_subdir = categorical_masks_dir / image_name
    assert categorical_masks_subdir.exists(), f"Subdir {categorical_masks_subdir} does not exist."
    assert len(list(categorical_masks_subdir.iterdir())) == 1, f"Subdir {categorical_masks_subdir} contains more than one mask."
    return list(categorical_masks_subdir.iterdir())[0]
