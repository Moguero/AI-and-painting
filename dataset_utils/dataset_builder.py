import tensorflow as tf
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from dataset_utils.file_utils import timeit
from dataset_utils.image_utils import decode_image
from dataset_utils.masks_encoder import one_hot_encode_image_patch_masks
from dataset_utils.patches_generator import extract_patches_with_overlap
from dataset_utils.patches_sorter import get_patches_above_coverage_percent_limit


@timeit
def get_train_and_test_dataset(
    n_patches_limit: int,
    n_classes: int,
    batch_size: int,
    test_proportion: float,
    patch_coverage_percent_limit: int,
    patches_dir_path: Path,
) -> (tf.data.Dataset, tf.data.Dataset):
    """

    :param n_patches_limit: Maximum number of patches used for the training.
    :param n_classes: Total number of classes, background not included.
    :param batch_size: Size of the batch.
    :param test_proportion: Float, used to set the proportion of the test dataset.
    :param patch_coverage_percent_limit: Int, minimum coverage percent of a patch labels on this patch.
    :param patches_dir_path: Path of the folder with patches of size patch_size.
    :return: A tuple of :
        - a training Dataset object of length (n_patches_limit // batch_size) * (1 - test_proportion),
        with tuples of tensors of size (batch_size, patch_size, patch_size, 3) and
        (batch_size, patch_size, patch_size, n_classes + 1) respectfully.
        - a test Dataset object of length (n_patches_limit // batch_size) * test_proportion,
        with tuples of tensors of size (batch_size, patch_size, patch_size, 3) and
        (batch_size, patch_size, patch_size, n_classes + 1) respectfully.
    """
    assert (
        0 <= test_proportion < 1
    ), f"Test proportion must be between 0 and 1 : {test_proportion} was given."
    logger.info("\nStart to build dataset...")
    assert (n_patches_limit // batch_size) * (
        1 - test_proportion
    ) >= 1, f"Size of training dataset is 0. Increase the n_patches_limit parameter or decrease the batch_size."

    # Get the paths of the valid patches for training
    image_patches_paths = get_patches_above_coverage_percent_limit(
        coverage_percent_limit=patch_coverage_percent_limit,
        patches_dir=patches_dir_path,
        n_patches_limit=n_patches_limit,
    )
    if n_patches_limit < len(image_patches_paths):
        logger.info(
            f"\nLimit of patches number set to {n_patches_limit} : only taking {n_patches_limit}/{len(image_patches_paths)} patches."
        )
        image_patches_paths = image_patches_paths[:n_patches_limit]

    # Create image patches tensors and their associated one-hot-encoded masks tensors
    image_tensors = [
        decode_image(Path(image_path))
        for image_path in tqdm(image_patches_paths, desc="Loading image tensors")
    ]
    mask_tensors = [
        one_hot_encode_image_patch_masks(
            image_patch_path=Path(image_patch_path), n_classes=n_classes
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
def build_predictions_dataset(
    target_image_tensor: tf.Tensor, patch_size: int, patch_overlap: int
) -> tf.data.Dataset:
    """
    Build a dataset of patches to make predictions on each one of them.

    :param target_image_tensor: Image to make predictions on.
    :param patch_size: Size of the patch.
    :param patch_overlap: Number of pixels on which neighbors patches intersect each other.
    :return: A Dataset object with tensors of size (1, patch_size, patch_size, 3). Its length corresponds of the number of patches generated.
    """
    logger.info("\nSlice the image into patches...")
    image_patches_tensors: list = extract_patches_with_overlap(
        image_tensor=target_image_tensor,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )
    prediction_dataset = tf.data.Dataset.from_tensor_slices(image_patches_tensors)
    prediction_dataset = prediction_dataset.batch(batch_size=1, drop_remainder=True)
    logger.info(f"\n{len(prediction_dataset)} patches created successfully.")
    return prediction_dataset


def get_dataset_iterator(dataset: tf.data.Dataset) -> tf.data.Iterator:
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    return iterator


def get_image_patches_paths(
    patches_dir_path: Path,
    n_patches_limit: int,
    batch_size: int,
    patch_coverage_percent_limit: int,
    test_proportion: float,
) -> [Path]:
    """
    Get images patches paths on which the model will train on.
    Also filter the valid patches above the coverage percent limit.

    :param patches_dir_path: Path of the patches root folder.
    :param n_patches_limit: Maximum number of patches used for the training.
    :param batch_size: Size of the batches.
    :param patch_coverage_percent_limit: Int, minimum coverage percent of a patch labels on this patch.
    :param test_proportion: Float, used to set the proportion of the test dataset.
    :return: A list of paths of images to train on.
    """
    assert (
        0 <= test_proportion < 1
    ), f"Test proportion must be between 0 and 1 : {test_proportion} was given."
    logger.info("\nStart to build dataset...")
    assert (n_patches_limit // batch_size) * (
        1 - test_proportion
    ) >= 1, f"Size of training dataset is 0. Increase the n_patches_limit parameter or decrease the batch_size."

    # Get the paths of the valid patches for training
    logger.info("\nGet the paths of the valid patches for training...")
    image_patches_paths = get_patches_above_coverage_percent_limit(
        coverage_percent_limit=patch_coverage_percent_limit,
        patches_dir=patches_dir_path,
        n_patches_limit=n_patches_limit,
    )
    if n_patches_limit < len(image_patches_paths):
        logger.info(
            f"\nLimit of patches number set to {n_patches_limit} : only taking {n_patches_limit}/{len(image_patches_paths)} patches."
        )
        image_patches_paths = image_patches_paths[:n_patches_limit]

    return image_patches_paths


def dataset_generator(
    image_patches_paths: [Path],
    n_classes: int,
    batch_size: int,
    test_proportion: float,
    stream: str,
) -> None:
    """
    Create a dataset generator that will be used to feed model.fit() (for stream == "train")
    and model.evaluate() (for stream == "test"). See stream parameter below.

    :param image_patches_paths: Paths of the images (already filtered) to train on.
    :param n_classes: Number of classes, background not included.
    :param batch_size: Size of the batches.
    :param test_proportion: Float, used to set the proportion of the test dataset.
    Splits the paths list in 2 : images paths used for training, the other ones for validation.
    :param stream: Str, equals to "train" or "test" :
        - stream == "train" : create infinite generator yielding batches (of size batch_size)
        of image/labels tensors tuple. Acts like "drop_remainder == True".
        - stream == "test" :
    """
    # Create image patches iterators and their associated one-hot-encoded masks tensors
    # & split the dataset according to its stream
    assert (
        stream == "train" or stream == "test"
    ), f"\nStream must be set 'train' or 'test' : set to {stream} here."

    train_limit_idx = int(len(image_patches_paths) * (1 - test_proportion))

    # Case of training dataset
    logger.info(
        f"\n{train_limit_idx}/{len(image_patches_paths)} patches taken for training."
        f"\n{(train_limit_idx // batch_size) * batch_size}/{train_limit_idx} patches will be kept and {train_limit_idx % batch_size}/{train_limit_idx} will be dropped.",
    )
    if stream == "train":
        while True:
            n_batches = train_limit_idx // batch_size
            for n_batch in range(n_batches):
                image_tensors = list()
                labels_tensors = list()
                for image_patch_path in image_patches_paths[n_batch * batch_size: (n_batch + 1) * batch_size]:
                    image_tensor, labels_tensor = decode_image(
                        Path(image_patch_path)
                    ), one_hot_encode_image_patch_masks(
                        image_patch_path=Path(image_patch_path), n_classes=n_classes
                    )
                    image_tensors.append(image_tensor)
                    labels_tensors.append(labels_tensor)
                yield tf.stack(image_tensors), tf.stack(labels_tensors)

    # Case of test dataset
    else:
        while True:
            n_batches = (len(image_patches_paths) - train_limit_idx) // batch_size  # to be checked later...
            for n_batch in range(n_batches):
                image_tensors = list()
                labels_tensors = list()
                for image_patch_path in image_patches_paths[train_limit_idx + n_batch * batch_size:  train_limit_idx + (n_batch + 1) * batch_size]:
                    image_tensor, labels_tensor = decode_image(
                        Path(image_patch_path)
                    ), one_hot_encode_image_patch_masks(
                        image_patch_path=Path(image_patch_path), n_classes=n_classes
                    )
                    image_tensors.append(image_tensor)
                    labels_tensors.append(labels_tensor)
                yield tf.stack(image_tensors), tf.stack(labels_tensors)
