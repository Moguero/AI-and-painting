import tensorflow as tf
from pathlib import Path

IMAGE_PATH = Path("/files/images/_DSC0043/_DSC0043.JPG")
IMAGE_PATHS = [
    Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043/_DSC0043.JPG"),
    Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0061/_DSC0061.JPG"),
]
MASK_PATHS = [
    Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/_DSC0043/feuilles_vertes/mask__DSC0043_feuilles_vertes__3466c2cda646448fbe8f4927f918e247.png"),
    Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/_DSC0061/feuilles_vertes/mask__DSC0061_feuilles_vertes__eef687829eb641c59f63ad80199b0de0.png"),
]
IMAGE_TYPE = "JPG"
BATCH_SIZE = 1

# enables eager execution
tf.compat.v1.enable_eager_execution()


# todo : implement a try/except to decode only jpg or png
def decode_image(file_path: Path, image_type: str, channels=3) -> tf.Tensor:
    """Turns a png or jpeg image into its tensor version."""
    value = tf.io.read_file(file_path)
    if image_type == "png" or image_type == "PNG":
        decoded_image = tf.image.decode_png(value, channels=channels)
    elif (
        image_type == "jpg"
        or image_type == "JPG"
        or image_type == "jpeg"
        or image_type == "JPEG"
    ):
        decoded_image = tf.image.decode_jpeg(value, channels=channels)
    else:
        decoded_image = tf.image.decode_image(value, channels=channels)
    return decoded_image


# todo : mask_path should be one-hot encoded and not rgb
def get_dataset(image_paths: [Path], mask_paths: [Path], image_type: str, batch_size: int):
    """We first create a 1D dataset of image_name/mask_name tensors, which we next map to an image dataset by decoding the paths.
    We also split the dataset into batches."""
    image_paths_tensor = tf.constant(image_paths)
    mask_paths_tensor = tf.constant(mask_paths)
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths_tensor, mask_paths_tensor)
    )

    def _parse_function(image_path: Path, mask_path: Path):
        return decode_image(file_path=image_path, image_type=image_type), decode_image(
            file_path=mask_path, image_type=image_type
        )

    map_dataset = dataset.map(_parse_function)
    dataset = map_dataset.batch(batch_size=batch_size, drop_remainder=False)
    return dataset


def get_dataset_iterator(dataset: tf.data.Dataset):
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    return iterator


dataset = get_dataset(IMAGE_PATHS, MASK_PATHS, IMAGE_TYPE, BATCH_SIZE)
iterator = get_dataset_iterator(dataset)


# todo : sort images/masks paths in the right order to be sure the good mask is associated with the good image \
# by putting every pair image/mask in the files/dataset/<image_name> folder with 2 subfolders image/ and mask/
