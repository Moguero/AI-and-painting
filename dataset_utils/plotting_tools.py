import imageio
import random
import matplotlib.pyplot as plt
import numpy as np
from dataset_utils.mask_utils import load_mask, get_image_masks_paths
from pathlib import Path

MASK_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/AVZD6310/mask_AVZD6310_Person.png")
IMAGE_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043/_DSC0043.JPG")
MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/")


# todo : check that the mask is one of the image designated from image_path
def plot_mask_with_color(image_path: Path, mask_path: Path):
    """Plot a mask with transparent color and with the associated picture in the background."""
    plt.imshow(imageio.imread(image_path))
    mask = load_mask(mask_path)
    color = [1, 0, 0, 0.5]  # RED
    color_ndarray = np.array([[color] * mask.shape[1]] * mask.shape[0])
    img = np.multiply(mask, color_ndarray).astype(int)
    plt.imshow(img)
    plt.show()


# todo : optimize this plotting tool with tensorflow
def plot_colored_masks_with_background_image(image_path: Path, masks_dir: Path):
    """Plot all the masks of an image with a random color, with the image in the background."""
    plt.imshow(imageio.imread(image_path))
    image_masks_paths = get_image_masks_paths(image_path, masks_dir)
    image_masks = [load_mask(mask_path) for mask_path in image_masks_paths]
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
