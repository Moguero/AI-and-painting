from dataset_utils.mask_loader import get_image_masks_first_channel, load_mask
import numpy as np
from pathlib import Path

IMAGE_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043/_DSC0043.JPG")
MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks")


# todo : maybe delete this file

def stack_image_masks(image_path: Path, masks_dir: Path):
    """Creates a unique one-hot encoded mask by stacking the image masks."""
    image_masks = get_image_masks_first_channel(image_path, masks_dir)
    one_hot_encoded_masks = [one_hot_encode_mask(mask) for mask in image_masks]
    stacked_mask = np.stack(one_hot_encoded_masks, axis=2)
    return stacked_mask


# todo : implement the one-encoding algorithm with tensorflow
def one_hot_encode_mask(mask: np.ndarray):
    return np.where(mask == 255, 1, 0)


# todo : transform all the numpy array to tensor directly
