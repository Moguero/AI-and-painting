import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from dataset_utils.image_utils import decode_image
from dataset_utils.masks_encoder import (
    one_hot_encode_image_masks,
    one_hot_encode_image_patch_masks,
)

IMAGE_PATH = Path("/files/images/_DSC0043/_DSC0043.JPG")
IMAGE_PATHS = [
    Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/1/1.jpg"),
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
BATCH_SIZE = 5
N_CLASSES = 9
N_PATCHES = 10
CATEGORICAL_MASKS_DIR = Path(
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/categorical_masks"
)
PATCHES_DIR = Path(r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\patches")
TEST_PROPORTION = 0.2

# enables eager execution
tf.compat.v1.enable_eager_execution()


# todo : create function that creates image_paths and its corresponding one-hot encoded mask_paths
def get_dataset(
    image_paths: [Path],
    categorical_masks_dir: Path,
    n_classes: int,
    batch_size: int,
) -> tf.data.Dataset:
    """We first create a 1D dataset of image_name/mask_name tensors, which we next map to an images dataset by decoding the paths.
    We also split the dataset into batches."""
    image_tensors = [decode_image(image_path) for image_path in image_paths]
    mask_tensors = [
        one_hot_encode_image_masks(
            image_path=image_path,
            categorical_masks_dir=categorical_masks_dir,
            n_classes=n_classes,
        )
        for image_path in image_paths
    ]
    dataset = tf.data.Dataset.from_tensor_slices((image_tensors, mask_tensors))
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    return dataset


def get_dataset_2(
    image_paths: [Path],
    categorical_masks_dir: Path,
    n_classes: int,
) -> [tf.Tensor]:
    image_tensors = [decode_image(image_path) for image_path in image_paths]
    mask_tensors = [
        one_hot_encode_image_masks(
            image_path=image_path,
            categorical_masks_dir=categorical_masks_dir,
            n_classes=n_classes,
        )
        for image_path in image_paths
    ]
    return image_tensors, mask_tensors


def get_dataset_3(
    image_paths: [Path],
    categorical_masks_dir: Path,
    n_classes: int,
) -> [tf.Tensor]:
    image_tensors = [
        tf.expand_dims(decode_image(image_path), axis=0) for image_path in image_paths
    ]
    mask_tensors = [
        tf.expand_dims(
            one_hot_encode_image_masks(
                image_path=image_path,
                categorical_masks_dir=categorical_masks_dir,
                n_classes=n_classes,
            ),
            axis=0,
        )
        for image_path in image_paths
    ]
    return image_tensors, mask_tensors


def get_dataset_4(
    patches_dir: Path,
    n_classes: int,
) -> [tf.Tensor]:
    image_patches_paths = [
        image_path
        for image_subdir in patches_dir.iterdir()
        for patch_subdir in image_subdir.iterdir()
        for image_path in (patch_subdir / "image").iterdir()
    ]
    image_tensors = [
        tf.expand_dims(decode_image(image_path), axis=0)
        for image_path in tqdm(image_patches_paths, desc="Loading image tensors")
    ]
    mask_tensors = [
        tf.expand_dims(
            one_hot_encode_image_patch_masks(
                image_patch_path=image_patch_path,
                n_classes=n_classes,
            ),
            axis=0,
        )
        for image_patch_path in tqdm(image_patches_paths, desc="Loading mask tensors")
    ]
    dataset = tf.data.Dataset.from_tensor_slices((image_tensors, mask_tensors))
    return dataset


def get_dataset_iterator(dataset: tf.data.Dataset) -> tf.data.Iterator:
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    return iterator


def main():
    dataset = get_dataset(IMAGE_PATHS, CATEGORICAL_MASKS_DIR, N_CLASSES, BATCH_SIZE)
    iterator = get_dataset_iterator(dataset)
    return dataset, iterator


# todo : sort images/masks paths in the right order to be sure the good mask is associated with the good images \
# by putting every pair images/mask in the files/dataset/<image_name> folder with 2 subfolders images/ and mask/


# get_dataset(IMAGE_PATHS, CATEGORICAL_MASKS_DIR, N_CLASSES, BATCH_SIZE)


# ---------
# DEBUG


def get_small_dataset(
    patches_dir: Path,
    n_patches: int,
    n_classes: int,
    batch_size: int,
    test_proportion: float,
) -> [tf.Tensor]:
    assert 0 <= test_proportion < 1, f"Test proportion must be between 0 and 1 : {test_proportion} was given."

    image_patches_paths = [
        image_path
        for image_subdir in patches_dir.iterdir()
        for patch_subdir in image_subdir.iterdir()
        for image_path in (patch_subdir / "image").iterdir()
    ]
    image_patches_paths = image_patches_paths[:n_patches]
    image_tensors = [
        decode_image(image_path)
        for image_path in tqdm(image_patches_paths, desc="Loading image tensors")
    ]
    mask_tensors = [
        one_hot_encode_image_patch_masks(
            image_patch_path=image_patch_path,
            n_classes=n_classes,
        )
        for image_patch_path in tqdm(image_patches_paths, desc="Loading mask tensors")
    ]
    # dataset = tf.data.Dataset.from_tensor_slices((image_tensors, mask_tensors))
    train_limit_idx = int(n_patches * (1 - test_proportion))
    X_train = tf.stack(image_tensors[: train_limit_idx], axis=0)
    y_train = tf.stack(mask_tensors[: train_limit_idx], axis=0)
    X_test = tf.stack(image_tensors[train_limit_idx:], axis=0)
    y_test = tf.stack(mask_tensors[train_limit_idx:], axis=0)

    # X_train = tf.split(X_train, num_or_size_splits=len(X_train) // batch_size, axis=0)
    # y_train = tf.split(y_train, num_or_size_splits=len(y_train) // batch_size, axis=0)
    # X_test = tf.split(X_test, num_or_size_splits=len(X_test) // batch_size, axis=0)
    # y_test = tf.split(y_test, num_or_size_splits=len(y_test) // batch_size, axis=0)

    return X_train, X_test, y_train, y_test


# todo : delete all the non usable patches before
def get_small_dataset_2(
    patches_dir: Path,
    n_patches: int,
    n_classes: int,
    batch_size: int,
    test_proportion: float,
) -> [tf.Tensor]:
    assert 0 <= test_proportion < 1, f"Test proportion must be between 0 and 1 : {test_proportion} was given."

    image_patches_paths = [
        image_path
        for image_subdir in patches_dir.iterdir()
        for patch_subdir in image_subdir.iterdir()
        for image_path in (patch_subdir / "image").iterdir()
    ]
    image_patches_paths = image_patches_paths[:n_patches]
    image_tensors = [
        decode_image(image_path)
        for image_path in tqdm(image_patches_paths, desc="Loading image tensors")
    ]
    mask_tensors = [
        one_hot_encode_image_patch_masks(
            image_patch_path=image_patch_path,
            n_classes=n_classes,
        )
        for image_patch_path in tqdm(image_patches_paths, desc="Loading mask tensors")
    ]
    dataset = tf.data.Dataset.from_tensor_slices((image_tensors, mask_tensors))
    batch_dataset = dataset.batch(batch_size, drop_remainder=True)
    return batch_dataset


# X_train, X_test, y_train, y_test = get_small_dataset(PATCHES_DIR, N_PATCHES, N_CLASSES, BATCH_SIZE, TEST_PROPORTION)
# get_small_dataset_2(PATCHES_DIR, N_PATCHES, N_CLASSES, BATCH_SIZE, TEST_PROPORTION)
