import imageio
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from label_utils.mask_loader import load_mask, get_image_masks

MASK_PATH = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/AVZD6310/mask_AVZD6310_Person.png"
IMAGE_PATH = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/AVZD6310/AVZD6310.png"
LABEL_MASK_DIR = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/"


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


def plot_colored_masks_with_background_image(image_path: str, label_mask_dir: str):
    """Plot all the masks of an image with a random color, with the image in the background."""
    plt.imshow(imageio.imread(image_path))
    image_masks = get_image_masks(image_path, label_mask_dir)
    for mask in image_masks:
        random_color = [random.random(), random.random(), random.random(), 0.5]
        color_ndarray = np.array([[random_color] * mask.shape[1]] * mask.shape[0])
        img = np.multiply(mask, color_ndarray).astype(int)
        plt.imshow(img)
    plt.axis("off")
    plt.show()


def transform_png_image_ndarray_to_dataframe(mask_path: str):
    img = load_mask(mask_path)
    multi_index = pd.MultiIndex.from_product([range(s) for s in img.shape], names=["x", "y", "channel"])
    img_df = pd.DataFrame({"img": img.flatten()}, index=multi_index).unstack("channel").rename(columns={0: "R", 1: "G", 2: "B", 3:"A"})
    return img_df



# todo : in the future, make a mapping between the class of a mask and the color that it has to have on the picture (ie a color palette)


# TODO : typer tous les inputs outputs de fonctions
# todo : documenter toutes les fonctions
# TODO : transformer les types str en Path quand c'est le cas
# todo : rename labels_masks_dir en masks_dir only
