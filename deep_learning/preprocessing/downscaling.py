from pathlib import Path

import numpy as np
from skimage.transform import downscale_local_mean

from constants import DOWNSCALE_FACTORS, IMAGE_PATH
from dataset_utils.image_utils import decode_image
from dataset_utils.plotting_tools import plot_image_from_array


def downscale_image(image_path: Path, downscale_factors: tuple) -> np.ndarray:
    image_array = decode_image(image_path)
    downscaled_array = downscale_local_mean(image_array, downscale_factors).astype(int)
    return downscaled_array


def plot_downscale_image(image_path: Path, downscale_factors: tuple) -> None:
    downscaled_array = downscale_image(image_path, downscale_factors)
    plot_image_from_array(downscaled_array)


# ------
# DEBUG
# downscale_image(IMAGE_PATH, DOWNSCALE_FACTORS)
# plot_downscale_image(IMAGE_PATH, DOWNSCALE_FACTORS)
