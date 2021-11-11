import tensorflow as tf

from pathlib import Path
from math import ceil
from skimage.transform import downscale_local_mean

from constants import DOWNSCALE_FACTORS, IMAGE_PATH
from dataset_utils.image_utils import decode_image
from dataset_utils.plotting_tools import plot_image_from_array


def downscale_image(image_path: Path, downscale_factors: tuple) -> tf.Tensor:
    """
    Downscale the image.

    :param image_path: Path of the image to downscale.
    :param downscale_factors: Factors on which we want to downscale the image.
    :return: A tensor of shape ceil(image_tensor.shape / downscale_factors).
    """
    image_tensor = decode_image(image_path)
    downscaled_array = downscale_local_mean(image=image_tensor, factors=downscale_factors).astype(int)
    downscaled_tensor = tf.constant(downscaled_array, dtype=tf.int32)

    # checks if the dimensions of the output are correct
    assert downscaled_tensor.shape[0] == ceil(
        image_tensor.shape[0] / downscale_factors[0]
    ), f"Downscaled tensor has height shape {downscaled_tensor.shape[0]} while it should have {ceil(image_tensor.shape[0] / downscale_factors[0])}"
    assert downscaled_tensor.shape[1] == ceil(
        image_tensor.shape[1] / downscale_factors[1]
    ), f"Downscaled tensor has width shape {downscaled_tensor.shape[1]} while it should have {ceil(image_tensor.shape[1] / downscale_factors[1])}"
    assert downscaled_tensor.shape[2] == ceil(
        image_tensor.shape[2] / downscale_factors[2]
    ), f"Downscaled tensor has channels shape {downscaled_tensor.shape[2]} while it should have {ceil(image_tensor.shape[2] / downscale_factors[2])}"

    return downscaled_tensor


def plot_downscale_image(image_path: Path, downscale_factors: tuple) -> None:
    downscaled_tensor = downscale_image(image_path, downscale_factors)
    downscaled_array = downscaled_tensor.numpy()
    plot_image_from_array(downscaled_array)


# ------
# DEBUG
# downscale_image(IMAGE_PATH, DOWNSCALE_FACTORS)
# plot_downscale_image(IMAGE_PATH, DOWNSCALE_FACTORS)
