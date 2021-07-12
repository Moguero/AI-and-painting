import imageio
import pandas as pd
import numpy as np
from scipy import misc

mask_path = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/masks/test/mask_angular_logo_lettre.png"
image_path = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/angular_logo.png"


def load_mask(mask_path: str) -> np.ndarray:
    """Turns png mask into numpy array"""
    mask_array = imageio.imread(mask_path)
    return mask_array


def load_mask_bis(mask_path: str):
    mask_array = imageio.imread(mask_path)
    return mask_array
