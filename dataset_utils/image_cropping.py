import tensorflow as tf

from constants import OUTPUT_PATH, TARGET_HEIGHT, TARGET_WIDTH
from dataset_utils.image_utils import decode_image, get_tensor_dims


def crop_tensor(
    tensor: tf.Tensor, target_height: int, target_width: int
) -> tf.Tensor:
    height_index, width_index, channels_index = get_tensor_dims(tensor)

    # center the crop
    height_difference = tensor.get_shape()[height_index] - target_height
    width_difference = tensor.get_shape()[width_index] - target_width
    assert height_difference >= 0
    if height_difference % 2 != 0:
        height_offset = height_difference // 2
    else:
        height_offset = height_difference // 2 + 1
    assert width_difference >= 0
    if width_difference % 2 != 0:
        width_offset = width_difference // 2
    else:
        width_offset = width_difference // 2 + 1

    # cropping and saving
    cropped_tensor = tf.image.crop_to_bounding_box(
        tensor, height_offset, width_offset, target_height, target_width
    )
    return cropped_tensor


def save_cropped_image(
    image_path: Path, output_path: Path, target_height: int, target_width: int
) -> None:
    image_tensor = decode_image(image_path)
    cropped_tensor = crop_tensor(image_tensor, target_height, target_width)
    encoded_image_tensor = tf.io.encode_jpeg(cropped_tensor, format='rgb', optimize_size=True)
    tf.io.write_file(filename=str(output_path), contents=encoded_image_tensor)


# -------
# DEBUG
# tensor = ...
# crop_tensor(..., OUTPUT_PATH, TARGET_HEIGHT, TARGET_WIDTH)
