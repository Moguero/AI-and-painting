import numpy as np
import tensorflow as tf
from pathlib import Path
from loguru import logger

from dataset_utils.image_utils import decode_image, get_tensor_dims


def smooth_predictions(
    predictions_path: Path, linearizer_kernel_size: int
) -> np.ndarray:
    """
    Make a mean of neighbors pixels.

    :param predictions_path: Path of the predictions image.
    :param linearizer_kernel_size: Size of the kernel on which to apply the mean.
    :return: A smoothed tensor of size (width // kernel_size, height // kernel_size)
    """
    logger.info("\nSmoothing predictions...")
    predictions_tensor = decode_image(predictions_path)[:, :, :3]
    height_index, width_index, channels_index = get_tensor_dims(predictions_tensor)
    n_rows = predictions_tensor.shape[height_index]
    n_columns = predictions_tensor.shape[width_index]

    smoothed_array = np.zeros(
        shape=(
            n_rows // linearizer_kernel_size,
            n_columns // linearizer_kernel_size,
            3,
        ),
        dtype=np.int32,
    )

    row_idx = 0
    n_sample_row = 0
    while row_idx + linearizer_kernel_size - 1 < n_rows:
        column_idx = 0
        n_sample_column = 0
        while column_idx + linearizer_kernel_size - 1 < n_columns:
            mean_rgb_value = list(
                get_dominant_rgb_value(
                    predictions_tensor[
                        row_idx : row_idx + linearizer_kernel_size,
                        column_idx : column_idx + linearizer_kernel_size,
                        :,
                    ],
                    linearizer_kernel_size,
                )
            )
            smoothed_array[n_sample_row, n_sample_column, :] = mean_rgb_value
            column_idx += linearizer_kernel_size
            n_sample_column += 1
        row_idx += linearizer_kernel_size
        n_sample_row += 1
    logger.info("\nPredictions have been smoothed successfully.")
    return smoothed_array


def get_dominant_rgb_value(sample: tf.Tensor, linearizer_kernel_size: int) -> tuple:
    count_rgb_values_dict = dict()

    for row_idx in range(len(sample[0])):
        for column_idx in range(len(sample[1])):
            rgb_value = tuple(sample[row_idx, column_idx].numpy())
            if rgb_value not in list(count_rgb_values_dict.keys()):
                count_rgb_values_dict[rgb_value] = 1
            else:
                count_rgb_values_dict[rgb_value] += 1
                if count_rgb_values_dict[rgb_value] == (
                    linearizer_kernel_size ** 2 // 2
                ):
                    return rgb_value
    dominant_rgb_value = max(count_rgb_values_dict, key=count_rgb_values_dict.get)
    return dominant_rgb_value


def save_smoothed_predictions(
    predictions_path: Path, linearizer_kernel_size: int, output_dir_path: Path
) -> None:
    predictions_array = smooth_predictions(predictions_path, linearizer_kernel_size)
    output_path = output_dir_path / "test2.png"
    tf.keras.preprocessing.image.save_img(output_path, predictions_array)
    logger.info(f"\nFull predictions plot successfully saved at : {output_path}")


# -------
# DEBUG

# smooth_predictions(PREDICTIONS_PATH, LINEARIZER_KERNEL_SIZE)
# save_smoothed_predictions(PREDICTIONS_PATH, LINEARIZER_KERNEL_SIZE, OUTPUT_DIR_PATH)
