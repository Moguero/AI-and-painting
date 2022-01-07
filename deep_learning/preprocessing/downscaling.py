import tensorflow as tf

from pathlib import Path
from math import ceil
from skimage.transform import downscale_local_mean

from constants import DOWNSCALE_FACTORS, IMAGE_PATH, TEST_IMAGES_DIR_PATH, TEST_IMAGES_PATHS_LIST, MAX_HEIGHT_PIXELS, \
    MAX_WIDTH_PIXELS, DOWNSCALED_TEST_IMAGES_DIR_PATH, MIN_HEIGHT_PIXELS, MIN_WIDTH_PIXELS
from dataset_utils.image_utils import decode_image, get_image_name_without_extension
from dataset_utils.masks_encoder import save_tensor_to_jpg
from dataset_utils.plotting_tools import plot_image_from_array


def downscale_image(image_path: Path, downscale_factors: (int, int, int)) -> tf.Tensor:
    """
    Downscale the image.

    :param image_path: Path of the image to downscale.
    :param downscale_factors: Factors on which we want to downscale the image.
    :return: A tensor of shape ceil(image_tensor.shape / downscale_factors).
    """
    image_tensor = decode_image(image_path=image_path)
    downscaled_array = downscale_local_mean(
        image=image_tensor, factors=downscale_factors
    ).astype(int)
    downscaled_tensor = tf.constant(downscaled_array, dtype=tf.uint8)

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


def plot_downscale_image(image_path: Path, downscale_factors: (int, int, int)) -> None:
    downscaled_tensor = downscale_image(image_path, downscale_factors)
    downscaled_array = downscaled_tensor.numpy()
    plot_image_from_array(downscaled_array)


def save_downscaled_image(
    image_path: Path, downscale_factors: (int, int, int), output_image_path: Path
) -> None:
    assert not output_image_path.exists(), f"Image path already exists : {output_image_path}"
    downscaled_tensor = downscale_image(
        image_path=image_path, downscale_factors=downscale_factors
    )
    save_tensor_to_jpg(tensor=downscaled_tensor, output_filepath=output_image_path)


def bulk_save_downscaled_images(
    target_images_paths_list: [Path],
    output_dir_path: Path,
    prefix_to_add: str,
    pixels_height_limit: int,
    pixels_widht_limit: int,
) -> None:
    for image_path in target_images_paths_list:
        height_downscale_factor = int(decode_image(image_path).shape[0] / pixels_height_limit) + 1
        width_downscale_factor = int(decode_image(image_path).shape[1] / pixels_widht_limit) + 1
        downscale_factor = max(height_downscale_factor, width_downscale_factor)

        downscale_factors = (downscale_factor, downscale_factor, 1)
        if not output_dir_path.exists():
            output_dir_path.mkdir(parents=True)
        output_image_path = output_dir_path / (
                prefix_to_add
                + get_image_name_without_extension(image_path=image_path)
                + ".jpg"
        )
        save_downscaled_image(
            image_path=image_path,
            downscale_factors=downscale_factors,
            output_image_path=output_image_path,
        )


# ------
# DEBUG
# downscale_image(IMAGE_PATH, DOWNSCALE_FACTORS)
# plot_downscale_image(IMAGE_PATH, DOWNSCALE_FACTORS)

# bulk_save_downscaled_images(TEST_IMAGES_PATHS_LIST, DOWNSCALED_TEST_IMAGES_DIR_PATH / "max", "downscaled_max_", MAX_HEIGHT_PIXELS, MAX_WIDTH_PIXELS)
# bulk_save_downscaled_images(TEST_IMAGES_PATHS_LIST, DOWNSCALED_TEST_IMAGES_DIR_PATH / "min", "downscaled_min_", MIN_HEIGHT_PIXELS, MIN_WIDTH_PIXELS)
