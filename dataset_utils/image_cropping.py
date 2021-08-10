from pathlib import Path
import tensorflow as tf

from dataset_utils.image_utils import decode_image

IMAGE_PATH = Path(
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0030/_DSC0030.jpg"
)
OUTPUT_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/test.jpg")

TARGET_HEIGHT = 2176
TARGET_WIDTH = 3264


def crop_image(
    image_path: Path, output_path: Path, target_height: int, target_width: int
) -> None:
    image_tensor = decode_image(image_path)

    # center the crop
    height_difference = image_tensor.get_shape()[0] - target_height
    width_difference = image_tensor.get_shape()[1] - target_width
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
        image_tensor, height_offset, width_offset, target_height, target_width
    )
    encoded_image_tensor = tf.io.encode_jpeg(cropped_tensor, format='rgb', optimize_size=True)
    tf.io.write_file(filename=str(output_path), contents=encoded_image_tensor)

# crop_image(IMAGE_PATH, OUTPUT_PATH, TARGET_HEIGHT, TARGET_WIDTH)