import tensorflow as tf
from loguru import logger
from tensorflow import keras

from constants import (
    N_CLASSES,
    INPUT_SHAPE,
    PATCH_SIZE,
    OPTIMIZER,
    LOSS_FUNCTION,
    METRICS,
    CHECKPOINT_ROOT_DIR_PATH,
    N_PATCHES_LIMIT,
    BATCH_SIZE,
    TEST_PROPORTION,
    PATCH_COVERAGE_PERCENT_LIMIT,
    N_EPOCHS,
    PATCHES_DIR_PATH,
    ENCODER_KERNEL_SIZE,
)
from dataset_utils.file_utils import timeit, get_formatted_time
from deep_learning.models.unet import build_small_unet
from dataset_utils.dataset_builder import (
    dataset_generator,
    get_image_patches_paths,
)
from pathlib import Path


@timeit
def train_model(
    n_classes: int,
    input_shape: int,
    patch_size: int,
    optimizer: keras.optimizers,
    loss_function: str,
    metrics: [str],
    checkpoint_root_dir_path: Path,
    n_patches_limit: int,
    batch_size: int,
    test_proportion: float,
    patch_coverage_percent_limit: int,
    epochs: int,
    patches_dir_path: Path,
    encoder_kernel_size: int,
):
    """
    Build the model, compile it, create a dataset iterator, train and model and save the trained model in callbacks.

    :param n_classes: Number of classes used for the model, background not included
    :param input_shape: Standard shape of the input images used for the training.
    :param patch_size: Size of the patches (should be equal to input_shape).
    :param optimizer:
    :param loss_function:
    :param metrics:
    :param checkpoint_root_dir_path: Path of the directory where the checkpoints are stored.
    :param n_patches_limit: Maximum number of patches used for the training.
    :param batch_size: Size of the batch.
    :param test_proportion: Float, used to set the proportion of the test dataset.
    :param patch_coverage_percent_limit: Int, minimum coverage percent of a patch labels on this patch.
    :param epochs: Number of epochs.
    :param patches_dir_path: Path of the main patches directory.
    :param encoder_kernel_size: Tuple of 2 integers, size of the encoder kernel.
    :return:
    """
    assert (
        input_shape == patch_size
    ), f"Input shape must be the same as the patch size, but patch size {patch_size} and input shape {input_shape} were given."
    # Define the model
    logger.info("\nStart to build model...")
    model = build_small_unet(n_classes, input_shape, batch_size, encoder_kernel_size)
    logger.info("\nModel built successfully.")

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    # Init the callbacks
    checkpoint_path = checkpoint_root_dir_path / get_formatted_time() / "cp.ckpt"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, verbose=1, save_weights_only=True
        )
    ]

    # todo : make a report of the classes used for the training
    # Init dataset
    # train_dataset, test_dataset = get_train_and_test_dataset(
    #     n_patches_limit=n_patches_limit,
    #     n_classes=n_classes,
    #     batch_size=batch_size,
    #     test_proportion=test_proportion,
    #     patch_coverage_percent_limit=patch_coverage_percent_limit,
    #     patches_dir_path=patches_dir_path,
    # )

    image_patches_paths = get_image_patches_paths(
        patches_dir_path,
        n_patches_limit,
        batch_size,
        patch_coverage_percent_limit,
        test_proportion,
    )

    train_dataset_iterator = dataset_generator(
        image_patches_paths=image_patches_paths,
        n_classes=n_classes,
        batch_size=batch_size,
        test_proportion=test_proportion,
        stream="train",
    )

    # todo : data augmentation with DataImageGenerator
    # Fit the model
    logger.info("\nStart model training...")
    # history = model.fit(train_dataset, epochs=epochs, callbacks=callbacks)
    # Warning : the steps_per_epoch param must be not null in order to end the infinite loop of the generator !
    history = model.fit(
        train_dataset_iterator,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=int(len(image_patches_paths) * (1 - test_proportion)) // batch_size,
    )
    logger.info("\nEnd of model training.")

    # Evaluate the model
    # loss, accuracy = model.evaluate(test_dataset, verbose=1)

    return model, history
    # return model, history, loss, accuracy


def load_saved_model(
    checkpoint_dir_path: Path,
    n_classes: int,
    input_shape: int,
    batch_size: int,
    encoder_kernel_size: int,
):
    logger.info("\nLoading the model...")
    model = build_small_unet(n_classes, input_shape, batch_size, encoder_kernel_size)
    filepath = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir_path)
    model.load_weights(filepath=filepath)
    # the warnings logs due to load_weights are here because we don't train (compile/fit) after : they disappear if we do
    logger.info("\nModel loaded successfully.")
    return model


# model, history = train_model(N_CLASSES, INPUT_SHAPE, PATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, METRICS, CHECKPOINT_ROOT_DIR_PATH, N_PATCHES_LIMIT, BATCH_SIZE, TEST_PROPORTION, PATCH_COVERAGE_PERCENT_LIMIT, N_EPOCHS, PATCHES_DIR_PATH, ENCODER_KERNEL_SIZE)
# train_model(N_CLASSES, INPUT_SHAPE, PATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, METRICS, CHECKPOINT_ROOT_DIR_PATH, N_PATCHES_LIMIT, BATCH_SIZE, TEST_PROPORTION, PATCH_COVERAGE_PERCENT_LIMIT, N_EPOCHS, PATCHES_DIR_PATH, ENCODER_KERNEL_SIZE)
