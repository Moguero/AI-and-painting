import random
import matplotlib.pyplot as plt
import tensorflow as tf

from dataset_utils.image_utils import decode_image
from dataset_utils.mask_utils import get_image_masks_paths
from pathlib import Path

MASK_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/_DSC0043/feuilles_vertes/mask__DSC0043_feuilles_vertes__3466c2cda646448fbe8f4927f918e247.png")
IMAGE_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043/_DSC0043.JPG")
MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/")


# todo : check that the mask is one of the image designated from image_path
def plot_mask_with_color(image_path: Path, mask_path: Path) -> None:
    """Plot a mask with transparent color and with the associated picture in the background."""
    image = decode_image(image_path).numpy()
    mask_tensor = tf.cast(decode_image(mask_path), float)
    color = [1, 0, 0, 0.5]  # RED
    color_tensor = tf.constant([[color] * mask_tensor.shape[1]] * mask_tensor.shape[0])
    mask = tf.multiply(color_tensor, mask_tensor).numpy().astype(int)
    plt.imshow(image)
    plt.imshow(mask)
    plt.axis("off")
    plt.show()


# todo : make a mapping between the class of a mask and the color that it has to have on the picture (ie a color palette)
def plot_colored_masks_with_background_image(image_path: Path, masks_dir: Path) -> None:
    """Plot all the masks of an image with a random color, with the image in the background."""
    image = decode_image(image_path).numpy()
    plt.imshow(image)
    image_masks_paths = get_image_masks_paths(image_path, masks_dir)
    for mask_path in image_masks_paths:
        mask_tensor = tf.cast(decode_image(mask_path), float)
        random_color = [random.random(), random.random(), random.random(), 0.5]
        color_tensor = tf.constant([[random_color] * mask_tensor.shape[1]] * mask_tensor.shape[0])
        mask = tf.multiply(color_tensor, mask_tensor).numpy().astype(int)
        plt.imshow(mask)
    plt.axis("off")
    plt.show()

# todo : documenter toutes les fonctions
