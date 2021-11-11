import tensorflow as tf
from pathlib import Path

from constants import TARGET_HEIGHT, TARGET_WIDTH
from dataset_utils.image_utils import decode_image, get_tensor_dims


def crop_tensor(tensor: tf.Tensor, target_height: int, target_width: int) -> tf.Tensor:
    """
    Returns a cropped tensor of size (target_height, target_width).
    First try to center the crop. If not possible, gives the most top-left part (see tests as example).

    :param tensor: The tensor to crop.
    :param target_height: Height of the output tensor.
    :param target_width: Width of the output tensor.
    :return: The cropped tensor of size (target_height, target_width).
    """
    height_index, width_index, channels_index = get_tensor_dims(tensor=tensor)

    # center the crop
    height_difference = tensor.get_shape()[height_index] - target_height
    width_difference = tensor.get_shape()[width_index] - target_width

    assert height_difference >= 0
    height_offset = height_difference // 2

    assert width_difference >= 0
    width_offset = width_difference // 2

    # cropping and saving
    cropped_tensor = tf.image.crop_to_bounding_box(
        image=tensor,
        offset_height=height_offset,
        offset_width=width_offset,
        target_height=target_height,
        target_width=target_width,
    )
    return cropped_tensor


def crop_patch_tensor(
        patch_tensor: tf.Tensor,
        patch_overlap: int,
) -> tf.Tensor:
    """

    :param patch_tensor: A tensor of size (x, y).
    :param patch_overlap: Number of pixels on which neighbors patches intersect each other.
    :return: A cropped tensor of size (x - patch_overlap, y - patch_overlap).
    """
    (
        patch_width_index,
        patch_height_index,
        patch_channels_index,
    ) = get_tensor_dims(tensor=patch_tensor)
    target_width = int(patch_tensor.shape[patch_width_index] - 2 * (patch_overlap / 2))
    target_height = int(
        patch_tensor.shape[patch_height_index] - 2 * (patch_overlap / 2)
    )
    patch_tensor = crop_tensor(
        tensor=patch_tensor, target_height=target_height, target_width=target_width
    )
    return patch_tensor


def save_cropped_image(
    image_path: Path, output_path: Path, target_height: int, target_width: int
) -> None:
    image_tensor = decode_image(file_path=image_path)
    cropped_tensor = crop_tensor(tensor=image_tensor, target_height=target_height, target_width=target_width)
    encoded_image_tensor = tf.io.encode_jpeg(
        cropped_tensor, format="rgb", optimize_size=True
    )
    tf.io.write_file(filename=str(output_path), contents=encoded_image_tensor)
