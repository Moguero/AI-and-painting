from label_utils.mask_loader import get_image_masks_first_channel
import numpy as np

IMAGE_PATH = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/AVZD6310.png"
LABEL_MASK_DIR = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/"


def stack_image_masks(image_path: str, labels_mask_dir: str):
    """Creates a unique one-hot encoded mask by stacking the image masks."""
    image_masks = get_image_masks_first_channel(image_path, labels_mask_dir)
    one_hot_encoded_masks = [one_hot_encode_mask(mask) for mask in image_masks]
    stacked_mask = np.stack(one_hot_encoded_masks, axis=2)
    return stacked_mask


def one_hot_encode_mask(mask: np.ndarray):
    return np.where(mask == 255, 1, 0)

# todo : transform all the numpy array to dataframe directly
