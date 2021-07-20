from dataset_utils.mask_utils import get_image_masks_paths
from pathlib import Path
import tensorflow as tf
from dataset_utils.dataset_builder import decode_image
from constants import MAPPING_CLASS_NUMBER

IMAGE_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043/_DSC0043.JPG")
MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks")
MASK_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/_DSC0043/feuilles_vertes/mask__DSC0043_feuilles_vertes__3466c2cda646448fbe8f4927f918e247.png")
IMAGE_TYPE = "png"
CHANNELS = 4
N_CLASSES = 9


# todo : get automatically the type of a picture with its path
# todo : remove the mask_type and channels parameter of the decode_image function + modify the hardcoded value below
def get_mask_first_channel(mask_path: Path) -> tf.Tensor:
    decoded_image = decode_image(mask_path, "png", 4)
    return decoded_image[:, :, 0]


def stack_image_masks(image_path: Path, masks_dir: Path) -> tf.Tensor:
    """Fetches the image mask first channels, transform it into a categorical tensor and add the categorical together."""
    image_masks_paths = get_image_masks_paths(image_path, masks_dir)
    shape = get_mask_first_channel(image_masks_paths[0]).shape
    stacked_tensor = tf.zeros(shape=shape, dtype=tf.int32)
    for mask_path in image_masks_paths:
        categorical_tensor = turn_mask_into_categorical_tensor(mask_path)
        stacked_tensor = tf.math.add(stacked_tensor, categorical_tensor)
    return stacked_tensor


# todo : remove hardcoded values 255 and 0
def turn_mask_into_categorical_tensor(mask_path: Path) -> tf.Tensor:
    tensor_first_channel = get_mask_first_channel(mask_path)
    mask_class = get_mask_class(mask_path)
    categorical_number = MAPPING_CLASS_NUMBER[mask_class]
    categorical_tensor = tf.where(tf.equal(tensor_first_channel, 255), categorical_number, 0)
    return categorical_tensor


def get_mask_class(mask_path: Path) -> str:
    return mask_path.parts[-2]


def one_hot_encode_image_masks(image_path: Path, masks_dir: Path, n_classes: int) -> tf.Tensor:
    stacked_masks_tensor = stack_image_masks(image_path, masks_dir)
    one_hot_encoded_tensor = tf.one_hot(stacked_masks_tensor, n_classes, dtype=tf.int32)
    return one_hot_encoded_tensor


# todo : transform all the numpy array to tensor directly
