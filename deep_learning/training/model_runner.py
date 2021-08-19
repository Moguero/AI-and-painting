import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from dataset_utils.file_utils import timeit, get_formatted_time
from dataset_utils.image_rebuilder import rebuild_predictions
from deep_learning.models.unet import build_small_unet
from dataset_utils.dataset_builder import get_dataset, build_predictions_dataset
from pathlib import Path


# DATA_DIR_ROOT = Path(r"/home/ec2-user/data")
DATA_DIR_ROOT = Path(r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files")

CHECKPOINT_ROOT_DIR_PATH = DATA_DIR_ROOT / "checkpoints"
CHECKPOINT_DIR_PATH = DATA_DIR_ROOT / "checkpoints/2021_08_12__17_33_34"
MODEL_PLOT_PATH = DATA_DIR_ROOT / "models\model.png"
SAVED_PATCHES_COVERAGE_PERCENT_PATH = DATA_DIR_ROOT / "temp_files/patches_coverage.csv"
ALL_MASKS_OVERLAP_INDICES_PATH = DATA_DIR_ROOT / "temp_files/all_masks_overlap_indices.csv"
# PATCHES_DIR_PATH = DATA_DIR_ROOT / "patches"
PATCHES_DIR_PATH = DATA_DIR_ROOT / "patches2"

BATCH_SIZE = 8
TEST_PROPORTION = 0.2
PATCH_COVERAGE_PERCENT_LIMIT = 75
N_CLASSES = 9
N_PATCHES_LIMIT = 10
INPUT_SHAPE = 256
# OPTIMIZER = "rmsprop"
OPTIMIZER = Adam(lr=1e-4)
LOSS_FUNCTION = "categorical_crossentropy"
METRICS = ["accuracy"]
EPOCHS = 2
PATCH_SIZE = 256


@timeit
def main(
        n_classes: int,
        input_shape: int,
        patch_size: int,
        optimizer,
        loss_function: str,
        metrics: [str],
        checkpoint_root_dir_path,
        n_patches_limit: int,
        batch_size: int,
        test_proportion: float,
        patch_coverage_percent_limit: int,
        epochs: int,
        patches_dir_path: Path
):
    assert input_shape == patch_size, f"Input shape must be the same as the patch size, but patch size {PATCH_SIZE} and input shape {INPUT_SHAPE} were given."
    # Define the model
    logger.info("\nStart to build model...")
    model = build_small_unet(n_classes, input_shape, patch_size)
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
    train_dataset, test_dataset = get_dataset(
        n_patches_limit=n_patches_limit,
        n_classes=n_classes,
        batch_size=batch_size,
        test_proportion=test_proportion,
        patch_coverage_percent_limit=patch_coverage_percent_limit,
        patches_dir_path=patches_dir_path,
    )
    breakpoint()

    # todo : use model.fit but with a generator instead of a Dataset
    # Fit the model
    logger.info("\nStart model training...")
    history = model.fit(train_dataset, epochs=epochs, callbacks=callbacks)
    logger.info("\nEnd of model training.")

    # # Evaluate the model
    # loss, accuracy = model.evaluate(test_dataset, verbose=1)

    breakpoint()


def load_saved_model(
    checkpoint_dir_path: Path, n_classes: int, input_shape: int, batch_size: int
):
    logger.info("\nLoading the model...")
    model = build_small_unet(n_classes, input_shape, batch_size)
    filepath = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir_path)
    model.load_weights(filepath=filepath)
    # the warnings logs due to load_weights are here because we don't train (compile/fit) after : they disappear if we do
    logger.info("\nModel loaded successfully.")
    return model


# predictions = make_predictions(TARGET_IMAGE_PATH, CHECKPOINT_DIR_PATH, INPUT_SHAPE, N_CLASSES, BATCH_SIZE)

# main(
#     N_CLASSES,
#     INPUT_SHAPE,
#     PATCH_SIZE,
#     OPTIMIZER,
#     LOSS_FUNCTION,
#     METRICS,
#     CHECKPOINT_ROOT_DIR_PATH,
#     N_PATCHES_LIMIT,
#     BATCH_SIZE,
#     TEST_PROPORTION,
#     PATCH_COVERAGE_PERCENT_LIMIT,
#     SAVED_PATCHES_COVERAGE_PERCENT_PATH,
#     EPOCHS,
#     ALL_MASKS_OVERLAP_INDICES_PATH
# )

# main(N_CLASSES, INPUT_SHAPE, PATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, METRICS, CHECKPOINT_ROOT_DIR_PATH, N_PATCHES_LIMIT, BATCH_SIZE, TEST_PROPORTION, PATCH_COVERAGE_PERCENT_LIMIT, EPOCHS, PATCHES_DIR_PATH)