import imageio
import random
import matplotlib.pyplot as plt
import numpy as np
from dataset_utils.mask_loader import load_mask, get_image_masks
from pathlib import Path

MASK_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/AVZD6310/mask_AVZD6310_Person.png")
IMAGE_PATH = Path("/files/images/_DSC0043/_DSC0043.JPG")
MASKS_DIR = Path("/files/labels_masks/")


# todo : check that the mask is one of the image designated from image_path
def plot_mask_with_color(image_path: str, mask_path: str):
    """Plot a mask with transparent color and with the associated picture in the background."""
    plt.imshow(imageio.imread(image_path))
    mask = load_mask(mask_path)
    color = [1, 0, 0, 0.5]  # RED
    color_ndarray = np.array([[color] * mask.shape[1]] * mask.shape[0])
    img = np.multiply(mask, color_ndarray).astype(int)
    plt.imshow(img)
    plt.show()


def plot_colored_masks_with_background_image(image_path: str, masks_dir: str):
    """Plot all the masks of an image with a random color, with the image in the background."""
    plt.imshow(imageio.imread(image_path))
    image_masks = get_image_masks(image_path, masks_dir)
    for mask in image_masks:
        random_color = [random.random(), random.random(), random.random(), 0.5]
        color_ndarray = np.array([[random_color] * mask.shape[1]] * mask.shape[0])
        img = np.multiply(mask, color_ndarray).astype(int)
        plt.imshow(img)
    plt.axis("off")
    plt.show()



# todo : in the future, make a mapping between the class of a mask and the color that it has to have on the picture (ie a color palette)


# TODO : typer tous les inputs outputs de fonctions
# todo : documenter toutes les fonctions
# TODO : transformer les types str en Path quand c'est le cas
