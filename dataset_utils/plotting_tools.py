import matplotlib.pyplot as plt
import tensorflow as tf

from constants import PALETTE_RGB
from dataset_utils.image_utils import decode_image, get_image_name_without_extension
from dataset_utils.image_utils import get_image_masks_paths, get_mask_class
from pathlib import Path

MASK_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/_DSC0043/feuilles_vertes/mask__DSC0043_feuilles_vertes__3466c2cda646448fbe8f4927f918e247.png")
IMAGE_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/1/1.jpg")
MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/all")
OUTPUT_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/test.jpg")


def plot_mask_with_color(image_path: Path, mask_path: Path) -> None:
    """Plot a mask with transparent color and with the associated picture in the background."""
    image_name = get_image_name_without_extension(image_path)
    assert image_name == mask_path.parts[-3], f"The mask {mask_path} doesn't match with the images {image_path}"
    image = decode_image(image_path).numpy()
    mask_tensor = tf.cast(decode_image(mask_path), float)
    color = [1, 0, 0, 0.5]  # RED
    color_tensor = tf.constant([[color] * mask_tensor.shape[1]] * mask_tensor.shape[0])
    mask = tf.multiply(color_tensor, mask_tensor).numpy().astype(int)
    plt.imshow(image)
    plt.imshow(mask)
    plt.axis("off")
    plt.show()


def plot_colored_masks_with_background_image(image_path: Path, masks_dir: Path) -> None:
    """Plot all the masks of an images with a random color, with the images in the background."""
    image = decode_image(image_path).numpy()
    plt.imshow(image)
    image_masks_paths = get_image_masks_paths(image_path, masks_dir)
    for mask_path in image_masks_paths:
        mask_tensor = tf.cast(decode_image(mask_path), float)
        mask_class = get_mask_class(mask_path)
        mask_color = list(PALETTE_RGB[mask_class]) + [0.5]
        color_tensor = tf.constant([[mask_color] * mask_tensor.shape[1]] * mask_tensor.shape[0])
        mask = tf.multiply(color_tensor, mask_tensor).numpy().astype(int)
        plt.imshow(mask)
    plt.axis("off")
    plt.show()


def save_colored_masks_with_background_image(image_path: Path, masks_dir: Path, output_path: Path) -> None:
    """Plot all the masks of an images with a random color, with the images in the background."""
    image = decode_image(image_path).numpy()
    plt.imshow(image)
    image_masks_paths = get_image_masks_paths(image_path, masks_dir)
    for mask_path in image_masks_paths:
        mask_tensor = tf.cast(decode_image(mask_path), float)
        mask_class = get_mask_class(mask_path)
        mask_color = list(PALETTE_RGB[mask_class]) + [0.5]
        color_tensor = tf.constant([[mask_color] * mask_tensor.shape[1]] * mask_tensor.shape[0])
        mask = tf.multiply(color_tensor, mask_tensor).numpy().astype(int)
        plt.imshow(mask)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight")


def plot_image_from_tensor(tensor: tf.Tensor) -> None:
    image = tensor.numpy()
    plt.imshow(image)
    plt.axis("off")
    plt.show()
