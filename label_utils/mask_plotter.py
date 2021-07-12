import imageio
import matplotlib.pyplot as plt
from label_utils.mask_loader import opacify_mask, load_mask, get_image_masks

MASK_PATH = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/AVZD6310/mask_AVZD6310_Person.png"
IMAGE_PATH = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/AVZD6310/AVZD6310.png"
LABEL_MASK_DIR = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/"


def plot_mask(mask_path: str) -> None:
    """Plot a mask by fully opacifying it first."""
    # img = opacify_mask(load_mask(mask_path))
    img = load_mask(mask_path)
    plt.imshow(img)
    plt.show()


# todo : plot all the masks on the same image, with different colors
# todo : function to plot both the image and its masks with transparency on the masks
def plot_masks_with_background_image(image_path: str, label_mask_dir: str):
    """Plot all the masks of an image, with the image in the background."""
    plt.imshow(imageio.imread(image_path))
    image_masks = get_image_masks(image_path, label_mask_dir)
    mask1 = image_masks[0]
    plt.imshow(mask1, cmap=plt.get_cmap('viridis'), alpha=0.5)
    mask2 = image_masks[1]
    plt.imshow(mask2, cmap=plt.get_cmap('plasma'), alpha=0.5)
    # for image_mask in image_masks:
    #     img = opacify_mask(image_mask)
    #     plt.imshow(img)
    plt.show()


# todo : get mask a random color i.e. turn the 255 pixels into the color and keep the (transparent?) background otherwise
def plot_mask_with_color(mask_path: str):
    img = load_mask(mask_path)
    plt.imshow(img)
    plt.show()



# todo : in the future, make a mapping between the class of a mask and the color that it has to have on the picture


# TODO : typer tous les inputs outputs de fonctions
# todo : documenter toutes les fonctions
# TODO : transformer les types str en Path quand c'est le cas
# todo : rename labels_masks_dir en masks_dir only

