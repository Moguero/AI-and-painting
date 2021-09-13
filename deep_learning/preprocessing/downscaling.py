from pathlib import Path

import numpy as np
from skimage.transform import downscale_local_mean

from dataset_utils.image_utils import decode_image


def downscale_image(image_path: Path, downscale_factors: tuple) -> np.ndarray:
    image_array = decode_image(image_path)
    downscaled_array = downscale_local_mean(image_array, downscale_factors)
    return downscaled_array
