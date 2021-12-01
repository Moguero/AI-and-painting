import tensorflow as tf
from pathlib import Path
from typing import Generator, Tuple

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

    return image_patches_paths


def train_dataset_generator(
    image_patches_paths: [Path],
    n_classes: int,
    batch_size: int,
    test_proportion: float,
    data_augmentation: bool = False,
) -> Generator[Tuple[tf.Tensor, tf.Tensor], None, None]:
    """
    Create a dataset generator that will be used to feed model.fit().
    This generator yields batches (acts like "drop_remainder == True") of augmented images.

    Warning : this function builds a generator which is only meant to be used with "model.fit()" function.
    Appropriate behaviour is not expected in a different context.

    :param image_patches_paths: Paths of the images (already filtered) to train on.
    :param n_classes: Number of classes, background not included.
    :param batch_size: Size of the batches.
    :param test_proportion: Float, used to set the proportion of the training images dataset.
    :param data_augmentation: Boolean, apply data augmentation to the each batch of the training dataset if True.
    :return: Yield 2 tensors of size (batch_size, patch_size, patch_size, 3) and (batch_size, patch_size, patch_size, n_classes + 1),
            corresponding to image tensors and their corresponding one-hot-encoded masks tensors
    """
    train_limit_idx = int(len(image_patches_paths) * (1 - test_proportion))

    # todo : data augmentation with ImageDataGenerator : will create a generator,
    #  that can be used to reload the augmented images in a tensor that will be yielded by this generator
    #  pass an augmentation_object parameter to this function (cf tuto)
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    # idée : data augmentation on each batch
    # Case of training dataset
    logger.info(
        f"\n{train_limit_idx}/{len(image_patches_paths)} patches taken for training (test proportion of {test_proportion})"
        f"\n{(train_limit_idx // batch_size) * batch_size}/{train_limit_idx} patches will be kept and {train_limit_idx % batch_size}/{train_limit_idx} will be dropped (drop remainder).",
    )

    while True:
        n_batches = train_limit_idx // batch_size
        for n_batch in range(n_batches):
            # list of length batch_size, containing image tensors of shape (256, 256, 3)
            image_tensors_list = list()
            # list of length batch_size, containing corresponding labels tensors of shape (256, 256, 10)
            labels_tensors_list = list()

            for image_patch_path in image_patches_paths[
                n_batch * batch_size : (n_batch + 1) * batch_size
            ]:
                image_tensor, labels_tensor = decode_image(
                    file_path=image_patch_path
                ), one_hot_encode_image_patch_masks(
                    image_patch_path=image_patch_path, n_classes=n_classes
                )
                image_tensors_list.append(image_tensor)
                labels_tensors_list.append(labels_tensor)

            if data_augmentation:
                augmented_image_tensors, augmented_labels_tensors = augment_batch(
                    image_tensors=image_tensors_list,
                    labels_tensors=labels_tensors_list,
                    batch_size=batch_size,
                )
            else:
                augmented_image_tensors, augmented_labels_tensors = tf.stack(
                    image_tensors_list
                ), tf.stack(labels_tensors_list)

            yield augmented_image_tensors, augmented_labels_tensors


# todo : test and debug this in model.evaluate()
def test_dataset_generator(
    image_patches_paths: [Path],
    n_classes: int,
    batch_size: int,
    test_proportion: float,
) -> Generator[Tuple[tf.Tensor, tf.Tensor], None, None]:
    """
    Create a dataset generator that will be used to feed model.evaluate()
    . This generator yields batches (acts like "drop_remainder == True") of augmented images.

    Warning : this function builds a generator which is only meant to be used with "model.evaluate()" function.
    Appropriate behaviour is not expected in a different context.

    :param image_patches_paths: Paths of the images (already filtered) to train on.
    :param n_classes: Number of classes, background not included.
    :param batch_size: Size of the batches.
    :param test_proportion: Float, used to set the proportion of the validation images dataset.
    :return: Yield 2 tensors of size (batch_size, patch_size, patch_size, 3) and (batch_size, patch_size, patch_size, n_classes + 1),
            corresponding to image tensors and their corresponding one-hot-encoded masks tensors
    """
    train_limit_idx = int(len(image_patches_paths) * (1 - test_proportion))

    while True:
        n_batches = (len(image_patches_paths) - train_limit_idx) // batch_size
        for n_batch in range(n_batches):
            # list of length batch_size, containing image tensors of shape (256, 256, 3)
            image_tensors_list = list()
            # list of length batch_size, containing corresponding labels tensors of shape (256, 256, 10)
            labels_tensors_list = list()

            for image_patch_path in image_patches_paths[
                train_limit_idx
                + n_batch * batch_size : train_limit_idx
                + (n_batch + 1) * batch_size
            ]:
                image_tensor, labels_tensor = decode_image(
                    file_path=image_patch_path
                ), one_hot_encode_image_patch_masks(
                    image_patch_path=image_patch_path, n_classes=n_classes
                )
                image_tensors_list.append(image_tensor)
                labels_tensors_list.append(labels_tensor)
            yield tf.stack(image_tensors_list), tf.stack(labels_tensors_list)


def augment_batch(
    image_tensors: [tf.Tensor],
    labels_tensors: [tf.Tensor],
    batch_size: int,
) -> (tf.Tensor, tf.Tensor):
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # rotation_range=20,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        horizontal_flip=True,
        # validation_split=0.2,
    )
    # todo : set up the fit method + data augmentation arguments
    # image_data_generator.fit(tf.stack(image_tensors))
    batches = 0
    # todo : rewrite this for loop properly
    for augmented_image_tensor, augmented_labels_tensor in image_data_generator.flow(
        x=tf.stack(image_tensors),
        y=tf.stack(labels_tensors),
        batch_size=batch_size,
        # shuffle=True,
    ):
        augmented_image_tensors = tf.constant(augmented_image_tensor, dtype=tf.int32)
        augmented_labels_tensors = tf.constant(augmented_labels_tensor, dtype=tf.int32)
        batches += 1
        if batches >= len(tf.stack(image_tensors)) / batch_size:
            break
    return augmented_image_tensors, augmented_labels_tensors
