import imageio
import numpy as np
import os

MASK_PATH = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/masks/test/mask_angular_logo_lettre.png"
IMAGE_PATH = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/angular_logo.png"


def load_mask(mask_path: str) -> np.ndarray:
    """Turns png mask into numpy ndarray"""
    mask_array = np.asarray(imageio.imread(mask_path))
    return mask_array


# todo : check if this is really useful : not sure...
def opacify_mask(mask_array: np.ndarray):
    """Delete the alpha channel of an array by deleting the transparency channel (4th channel).
    As a result, the mask will be fully opaque."""
    return mask_array[:, :, :3]


def get_image_name_without_extension(image_path: str):
    return image_path.split("/")[-1].split(".")[0]


def get_image_name_with_extension(image_path: str):
    return image_path.split("/")[-1]


def get_image_masks_paths(image_path: str, mask_dir: str):
    """Get the paths of the image masks.

    Remark: If the masks sub directory associated to the image does not exist,
    an AssertionError will be thrown."""
    image_masks_sub_dir = mask_dir + get_image_name_without_extension(image_path)
    assert os.path.exists(image_masks_sub_dir), f"Image masks sub directory {image_masks_sub_dir} does not exist"
    return [
        mask_dir + get_image_name_without_extension(image_path) + "/" + mask_name
        for mask_name in os.listdir(image_masks_sub_dir)
    ]


def get_image_masks(image_path: str, labels_masks_dir: str):
    image_masks_paths = get_image_masks_paths(image_path, labels_masks_dir)
    return [load_mask(mask_path) for mask_path in image_masks_paths]


def get_image_masks_first_channel(image_path: str, labels_masks_dir: str):
    image_masks_paths = get_image_masks_paths(image_path, labels_masks_dir)
    return [load_mask(mask_path)[:, :, 0] for mask_path in image_masks_paths]
