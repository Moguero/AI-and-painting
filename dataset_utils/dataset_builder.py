import tensorflow as tf
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from dataset_utils.file_utils import timeit
from dataset_utils.image_utils import decode_image
from dataset_utils.masks_encoder import one_hot_encode_image_patch_masks
from deep_learning.patches_generator import extract_image_patches
from deep_learning.patches_sorter import get_patches_above_coverage_percent_limit


MASKS_DIR = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files\labels_masks"
)
PATCHES_DIR = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files\patches"
)
SAVED_PATCHES_COVERAGE_PERCENT_PATH = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files\temp_files\patches_coverage.csv"
)
TARGET_IMAGE_PATH = Path(r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files\images\_DSC0048\_DSC0048.jpg")
PATCH_SIZE = 256
BATCH_SIZE = 2
N_CLASSES = 9
N_PATCHES = 10
TEST_PROPORTION = 0.2
PATCH_COVERAGE_PERCENT_LIMIT = 75


@timeit
def get_dataset(
    n_patches_limit: int,
    n_classes: int,
    batch_size: int,
    test_proportion: float,
    patch_coverage_percent_limit: int,
    saved_patches_coverage_percent_path: Path,
) -> [tf.Tensor]:
    assert (
        0 <= test_proportion < 1
    ), f"Test proportion must be between 0 and 1 : {test_proportion} was given."
    logger.info("\nStart to build dataset...")

    # Get the paths of the valid patches for training
    image_patches_paths = get_patches_above_coverage_percent_limit(
        coverage_percent_limit=patch_coverage_percent_limit,
        saved_patches_coverage_percent_path=saved_patches_coverage_percent_path,
    )
    if n_patches_limit < len(image_patches_paths):
        image_patches_paths = image_patches_paths[:n_patches_limit]

    # Create image patches tensors and their associated one-hot-encoded masks tensors
    image_tensors = [
        decode_image(Path(image_path))
        for image_path in tqdm(image_patches_paths, desc="Loading image tensors")
    ]
    mask_tensors = [
        one_hot_encode_image_patch_masks(
            image_patch_path=Path(image_patch_path),
            n_classes=n_classes,
        )
        for image_patch_path in tqdm(image_patches_paths, desc="Loading mask tensors")
    ]

    # Create a tf.data.Dataset object and batch it
    full_dataset = tf.data.Dataset.from_tensor_slices((image_tensors, mask_tensors))
    full_dataset = full_dataset.batch(batch_size, drop_remainder=True)
    # todo : shuffle the whole dataset

    # Split the dataset into a training and testing one
    train_limit_idx = int(len(full_dataset) * (1 - test_proportion))
    train_dataset = full_dataset.take(train_limit_idx)
    test_dataset = full_dataset.skip(train_limit_idx)

    logger.info("\nDataset built successfully.")
    return train_dataset, test_dataset


# todo : don't hardcode the batch_size to 1
def build_predictions_dataset(target_image_path: Path, patch_size: int) -> tf.data.Dataset:
    logger.info("\nSlice the image into patches...")
    image_patches_tensors: list = extract_image_patches(
        image_path=target_image_path,
        patch_size=patch_size,
    )
    prediction_dataset = tf.data.Dataset.from_tensor_slices(image_patches_tensors)
    prediction_dataset = prediction_dataset.batch(batch_size=1, drop_remainder=True)
    logger.info(f"\n{len(prediction_dataset)} patches created successfully.")
    return prediction_dataset


def get_dataset_generator(dataset: tf.data.Dataset) -> tf.data.Iterator:
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    return iterator


def get_train_and_test_dataset_iterators():
    train_dataset, test_dataset = get_dataset(
        n_patches_limit=N_PATCHES,
        n_classes=N_CLASSES,
        batch_size=BATCH_SIZE,
        test_proportion=TEST_PROPORTION,
        patch_coverage_percent_limit=PATCH_COVERAGE_PERCENT_LIMIT,
        saved_patches_coverage_percent_path=SAVED_PATCHES_COVERAGE_PERCENT_PATH,
    )
    train_generator, test_generator = get_dataset_generator(
        train_dataset
    ), get_dataset_generator(test_dataset)
    return train_generator, test_generator


# def get_small_dataset(
#     patches_dir: Path,
#     n_patches: int,
#     n_classes: int,
#     batch_size: int,
#     test_proportion: float,
# ) -> [tf.Tensor]:
#     assert 0 <= test_proportion < 1, f"Test proportion must be between 0 and 1 : {test_proportion} was given."
#
#     image_patches_paths = [
#         image_path
#         for image_subdir in patches_dir.iterdir()
#         for patch_subdir in image_subdir.iterdir()
#         for image_path in (patch_subdir / "image").iterdir()
#     ]
#     image_patches_paths = image_patches_paths[:n_patches]
#     image_tensors = [
#         decode_image(image_path)
#         for image_path in tqdm(image_patches_paths, desc="Loading image tensors")
#     ]
#     mask_tensors = [
#         one_hot_encode_image_patch_masks(
#             image_patch_path=image_patch_path,
#             n_classes=n_classes,
#         )
#         for image_patch_path in tqdm(image_patches_paths, desc="Loading mask tensors")
#     ]
#     # dataset = tf.data.Dataset.from_tensor_slices((image_tensors, mask_tensors))
#     train_limit_idx = int(n_patches * (1 - test_proportion))
#     X_train = tf.stack(image_tensors[: train_limit_idx], axis=0)
#     y_train = tf.stack(mask_tensors[: train_limit_idx], axis=0)
#     X_test = tf.stack(image_tensors[train_limit_idx:], axis=0)
#     y_test = tf.stack(mask_tensors[train_limit_idx:], axis=0)
#
#     # X_train = tf.split(X_train, num_or_size_splits=len(X_train) // batch_size, axis=0)
#     # y_train = tf.split(y_train, num_or_size_splits=len(y_train) // batch_size, axis=0)
#     # X_test = tf.split(X_test, num_or_size_splits=len(X_test) // batch_size, axis=0)
#     # y_test = tf.split(y_test, num_or_size_splits=len(y_test) // batch_size, axis=0)
#
#     return X_train, X_test, y_train, y_test


# get_dataset(IMAGE_PATHS, CATEGORICAL_MASKS_DIR, N_CLASSES, BATCH_SIZE)
# train, test = get_small_dataset_2(N_PATCHES, N_CLASSES, BATCH_SIZE, TEST_PROPORTION, PATCH_COVERAGE_PERCENT_LIMIT, SAVED_PATCHES_COVERAGE_PERCENT_PATH)
