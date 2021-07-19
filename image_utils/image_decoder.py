import tensorflow as tf

IMAGE_PATH = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043/_DSC0043.JPG"
IMAGE_PATHS = [
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043/_DSC0043.JPG",
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0061/_DSC0061.JPG"
]
MASK_PATHS = [
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/_DSC0043/feuilles_vertes/mask__DSC0043_feuilles_vertes__3466c2cda646448fbe8f4927f918e247.png",
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/_DSC0061/feuilles_vertes/mask__DSC0061_feuilles_vertes__eef687829eb641c59f63ad80199b0de0.png"
]
IMAGE_TYPE = 'JPG'
BATCH_SIZE = 1

# enables eager execution
tf.compat.v1.enable_eager_execution()


def decode_image(filename: str, image_type: str, channels=3):
    value = tf.io.read_file(filename)
    if image_type == 'png' or image_type == 'PNG':
        decoded_image = tf.image.decode_png(value, channels=channels)
    elif image_type == 'jpeg' or image_type == 'jpg' or image_type == 'JPEG' or image_type == 'JPG':
        decoded_image = tf.image.decode_jpeg(value, channels=channels)
    else:
        decoded_image = tf.image.decode_image(value, channels=channels)

    return decoded_image


# todo : mask_path should be one-hot encoded and not rgb
def get_dataset(image_paths: list, mask_paths: list, image_type: str, batch_size : int):
    """We first create a 1D dataset of image_name/mask_name tensors, which we next map to an image dataset by decoding the paths.
    We also split the dataset into batches."""
    image_paths_tensor = tf.constant(image_paths)
    mask_paths_tensor = tf.constant(mask_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, mask_paths_tensor))

    def _parse_function(image_path: str, mask_path: str):
        return decode_image(image_path, image_type), decode_image(mask_path, image_type)

    map_dataset = dataset.map(_parse_function)
    dataset = map_dataset.batch(batch_size=batch_size, drop_remainder=False)
    return dataset


# todo : seems to yield the first batch only
def get_final_input_tensor(dataset: tf.data.Dataset):
    """Yields the next batch"""
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    images, masks = iterator.get_next()
    return images, masks


### Debug

# dataset = get_dataset(IMAGE_PATHS, MASK_PATHS, IMAGE_TYPE, BATCH_SIZE)
# get_final_input_tensor(get_dataset(IMAGE_PATHS, MASK_PATHS, IMAGE_TYPE, BATCH_SIZE))
