from pathlib import Path
import tensorflow as tf

from dataset_utils.image_utils import decode_image


IMAGE_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/1/1.jpg")
SIZES = [1, 500, 500, 1]
STRIDES = [1, 1, 1, 1]
RATES = [1, 1, 1, 1]


def extract_image_patches(
    image_path: Path,
    sizes: list,
    strides: list,
    rates: list,
    padding: str = 'VALID',
):
    image = decode_image(image_path)
    image = tf.expand_dims(image, 0)
    breakpoint()
    tensor = tf.image.extract_patches(image, sizes=sizes, strides=strides, rates=rates, padding=padding)
    return tensor

# extract_image_patches(IMAGE_PATH, STRIDES, STRIDES, RATES)
