import tensorflow as tf
from pathlib import Path

from dataset_utils.image_utils import decode_image
from dataset_utils.mask_processing import one_hot_encode_image_masks

IMAGE_PATH = Path("/files/images/_DSC0043/_DSC0043.JPG")
IMAGE_PATHS = [
    Path(
        "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/1/1.jpg"
    ),
    Path(
        "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0130/_DSC0130.jpg"
    ),
]

# IMAGE_PATHS = [
#     Path(
#         "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/1/1.jpg"
#     ),
# ]


MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks")
BATCH_SIZE = 1
N_CLASSES = 9
CATEGORICAL_MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/categorical_masks")

# enables eager execution
tf.compat.v1.enable_eager_execution()


# todo : create function that creates image_paths and its corresponding one-hot encoded mask_paths
def get_dataset(
    image_paths: [Path],
    categorical_masks_dir: Path,
    n_classes:int,
    batch_size: int,
) -> tf.data.Dataset:
    """We first create a 1D dataset of image_name/mask_name tensors, which we next map to an image dataset by decoding the paths.
    We also split the dataset into batches."""
    image_tensors = [decode_image(image_path) for image_path in image_paths]
    mask_tensors = [one_hot_encode_image_masks(image_path=image_path, categorical_masks_dir=categorical_masks_dir, n_classes=n_classes) for image_path in image_paths]
    breakpoint()
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_tensors, mask_tensors)
    )
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    return dataset


def get_dataset_iterator(dataset: tf.data.Dataset) -> tf.data.Iterator:
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    return iterator


def main():
    dataset = get_dataset(IMAGE_PATHS, CATEGORICAL_MASKS_DIR, N_CLASSES, BATCH_SIZE)
    iterator = get_dataset_iterator(dataset)
    return dataset, iterator


# todo : sort images/masks paths in the right order to be sure the good mask is associated with the good image \
# by putting every pair image/mask in the files/dataset/<image_name> folder with 2 subfolders image/ and mask/
