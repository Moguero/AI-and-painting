import tensorflow as tf
from loguru import logger
from tensorflow import keras

from constants import N_CLASSES, INPUT_SHAPE, PATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, METRICS, CHECKPOINT_ROOT_DIR_PATH, \
    N_PATCHES_LIMIT, BATCH_SIZE, TEST_PROPORTION, PATCH_COVERAGE_PERCENT_LIMIT, N_EPOCHS, PATCHES_DIR_PATH, ENCODER_KERNEL_SIZE
from dataset_utils.file_utils import timeit, get_formatted_time
from deep_learning.models.unet import build_small_unet
from dataset_utils.dataset_builder import get_train_and_test_dataset
from pathlib import Path



@timeit
def train_model(
        n_classes: int,
        input_shape: int,
        patch_size: int,
        optimizer,
        loss_function: str,
        metrics: [str],
        checkpoint_root_dir_path: Path,
        n_patches_limit: int,
        batch_size: int,
        test_proportion: float,
        patch_coverage_percent_limit: int,
        epochs: int,
        patches_dir_path: Path,
        endoder_kernel_size: int
):
    assert input_shape == patch_size, f"Input shape must be the same as the patch size, but patch size {PATCH_SIZE} and input shape {INPUT_SHAPE} were given."
    # Define the model
    logger.info("\nStart to build model...")
    model = build_small_unet(n_classes, input_shape, batch_size, endoder_kernel_size)
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

    # Init dataset
    train_dataset, test_dataset = get_train_and_test_dataset(
        n_patches_limit=n_patches_limit,
        n_classes=n_classes,
        batch_size=batch_size,
        test_proportion=test_proportion,
        patch_coverage_percent_limit=patch_coverage_percent_limit,
        patches_dir_path=patches_dir_path,
    )

    # todo : use model.fit but with a generator instead of a Dataset ?
    # Fit the model
    logger.info("\nStart model training...")
    history = model.fit(train_dataset, epochs=epochs, callbacks=callbacks)
    logger.info("\nEnd of model training.")

    # Evaluate the model
    loss, accuracy = model.evaluate(test_dataset, verbose=1)

    return model, history, loss, accuracy


def load_saved_model(
    checkpoint_dir_path: Path, n_classes: int, input_shape: int, batch_size: int, encoder_kernel_size: int
):
    logger.info("\nLoading the model...")
    model = build_small_unet(n_classes, input_shape, batch_size, encoder_kernel_size)
    filepath = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir_path)
    model.load_weights(filepath=filepath)
    # the warnings logs due to load_weights are here because we don't train (compile/fit) after : they disappear if we do
    logger.info("\nModel loaded successfully.")
    return model


# train_model(N_CLASSES, INPUT_SHAPE, PATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, METRICS, CHECKPOINT_ROOT_DIR_PATH, N_PATCHES_LIMIT, BATCH_SIZE, TEST_PROPORTION, PATCH_COVERAGE_PERCENT_LIMIT, N_EPOCHS, PATCHES_DIR_PATH, ENCODER_KERNEL_SIZE)
