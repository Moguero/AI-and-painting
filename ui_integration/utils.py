import time
import numpy as np
import tensorflow as tf
from pathlib import Path


def get_formatted_time():
    return time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())


def decode_image(file_path: Path) -> tf.Tensor:
    """
    Turns a png or jpeg images into its tensor 3D version (dropping the 4th channel if PNG).

    :param filepath: The path of the file to decode.
    """
    value = tf.io.read_file(str(file_path))
    image_type = get_image_type(file_path)
    if image_type == "png" or image_type == "PNG":
        decoded_image = tf.image.decode_png(value, channels=3)
    elif (
        image_type == "jpg"
        or image_type == "JPG"
        or image_type == "jpeg"
        or image_type == "JPEG"
    ):
        decoded_image = tf.image.decode_jpeg(value, channels=3)
    else:
        raise ValueError(f"Image {file_path} if not png nor jpeg.")
    return decoded_image


def get_image_type(image_path: Path) -> str:
    return image_path.parts[-1].split(".")[-1]


def get_image_tensor_shape(image_tensor: tf.Tensor):
    """
    Get the height, width and channels number of an image.
    """
    height_index, width_index, channels_index = get_tensor_dims(tensor=image_tensor)

    image_height = image_tensor.shape[height_index]
    image_width = image_tensor.shape[width_index]
    if channels_index is not None:
        channels_number = image_tensor.shape[channels_index]
    else:
        channels_number = None

    return image_height, image_width, channels_number


def get_tensor_dims(tensor: tf.Tensor) -> tuple:
    n_dims = len(list(tensor.shape))
    if n_dims == 2:
        height_index = 0
        width_index = 1
        channels_index = None
    elif n_dims == 3:
        height_index = 0
        width_index = 1
        channels_index = 2
    elif n_dims == 4:
        height_index = 1
        width_index = 2
        channels_index = 3
    else:
        raise ValueError(f"Dimension is {n_dims} : expected 2, 3 or 4.")
    return height_index, width_index, channels_index


def get_image_name_without_extension(image_path: Path) -> str:
    return image_path.parts[-1].split(".")[0]


def get_file_name_with_extension(file_path: Path) -> str:
    return file_path.parts[-1]


def turn_2d_tensor_to_3d_tensor(
    tensor_2d: tf.Tensor,
) -> np.ndarray:
    """
    Turn a 2D tensor into a 3D array by tripling the same mask layer.

    :param tensor_2d: A 2D categorical tensor of size (width_size, height_size)
    :return: A 3D array of size (width_size, height_size, 3)
    """
    array_2d = tensor_2d.numpy()
    vectorize_function = np.vectorize(lambda x: (x, x, x))
    array_3d = np.stack(vectorize_function(array_2d), axis=2)
    return array_3d
