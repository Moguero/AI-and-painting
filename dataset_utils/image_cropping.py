import tensorflow as tf
from pathlib import Path

from dataset_utils.image_utils import (
    decode_image,
    get_image_tensor_shape,
)


def crop_tensor(tensor: tf.Tensor, target_height: int, target_width: int) -> tf.Tensor:
    """
    Returns a cropped tensor of size (target_height, target_width).
    First try to center the crop. If not possible, gives the most top-left part (see tests as example).

    :param tensor: The tensor to crop.
    :param target_height: Height of the output tensor.
    :param target_width: Width of the output tensor.
    :return: The cropped tensor of size (target_height, target_width).
    """
    tensor_height, tensor_width, channels_number = get_image_tensor_shape(
        image_tensor=tensor
    )

    # center the crop
    height_difference = tensor_height - target_height
    width_difference = tensor_width - target_width

    assert height_difference >= 0
    height_offset = int(height_difference // 2)

    assert width_difference >= 0
    width_offset = int(width_difference // 2)

    # cropping and saving
    cropped_tensor = tf.image.crop_to_bounding_box(
        image=tensor,
        offset_height=height_offset,
        offset_width=width_offset,
        target_height=target_height,
        target_width=target_width,
    )

    # Check that the output size is correct
    (
        cropped_tensor_height,
        cropped_tensor_width,
        channels_number,
    ) = get_image_tensor_shape(image_tensor=cropped_tensor)
    assert (cropped_tensor_height == target_height) and (
        cropped_tensor_height == target_width
    ), f"\nCropped tensor shape is ({cropped_tensor_height}, {cropped_tensor_width}) : should be {target_height}, {target_width}"

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
    assert (
        patch_overlap % 2 == 0
    ), f"Patch overlap argument must be a pair number. The one specified was {patch_overlap}."

    image_height, image_width, channels_number = get_image_tensor_shape(
        image_tensor=patch_tensor
    )
    target_width = int(image_width - 2 * (patch_overlap / 2))
    target_height = int(image_height - 2 * (patch_overlap / 2))

    patch_tensor = crop_tensor(
        tensor=patch_tensor, target_height=target_height, target_width=target_width
    )
    return patch_tensor


def save_cropped_image(
    image_path: Path, output_path: Path, target_height: int, target_width: int
) -> None:
    image_tensor = decode_image(file_path=image_path)
    cropped_tensor = crop_tensor(
        tensor=image_tensor, target_height=target_height, target_width=target_width
    )
    encoded_image_tensor = tf.io.encode_jpeg(
        cropped_tensor, format="rgb", optimize_size=True
    )
    tf.io.write_file(filename=str(output_path), contents=encoded_image_tensor)
