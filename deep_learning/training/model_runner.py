import time

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


CHECKPOINT_ROOT_DIR_PATH = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\checkpoints"
)
CHECKPOINT_DIR_PATH = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\checkpoints\2021_08_12__17_33_34"
)
MODEL_PLOT_PATH = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files\models\model.png"
)
SAVED_PATCHES_COVERAGE_PERCENT_PATH = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files\temp_files\patches_coverage.csv"
)
TARGET_IMAGE_PATH = Path(r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files\images\IMG_2304\IMG_2304.jpg")



BATCH_SIZE = 8
TEST_PROPORTION = 0.2
PATCH_COVERAGE_PERCENT_LIMIT = 75
N_CLASSES = 9
N_PATCHES_LIMIT = 100
INPUT_SHAPE = 256
# OPTIMIZER = "rmsprop"
OPTIMIZER = Adam(lr=1e-4)
LOSS_FUNCTION = "categorical_crossentropy"
METRICS = ["accuracy"]
EPOCHS = 2


@timeit
def main():
    # Define the model
    logger.info("\nStart to build model...")
    model = build_small_unet(N_CLASSES, INPUT_SHAPE, BATCH_SIZE)
    logger.info("\nModel built successfully.")

    # Compile the model
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)

    # Init the callbacks
    checkpoint_path = CHECKPOINT_ROOT_DIR_PATH / get_formatted_time() / "cp.ckpt"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, verbose=1, save_weights_only=True
        )
    ]

    # Init dataset
    train_dataset, test_dataset = get_dataset(
        n_patches_limit=N_PATCHES_LIMIT,
        n_classes=N_CLASSES,
        batch_size=BATCH_SIZE,
        test_proportion=TEST_PROPORTION,
        patch_coverage_percent_limit=PATCH_COVERAGE_PERCENT_LIMIT,
        saved_patches_coverage_percent_path=SAVED_PATCHES_COVERAGE_PERCENT_PATH,
    )
    breakpoint()

    # todo : use model.fit but with a generator instead of a Dataset
    # Fit the model
    logger.info("\nStart model training...")
    history = model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)
    logger.info("\nEnd of model training.")

    # # Evaluate the model
    # loss, accuracy = model.evaluate(test_dataset, verbose=1)

    breakpoint()


def make_predictions(
    target_image_path,
    checkpoint_dir_path: Path,
    patch_size: int,
    n_classes: int,
    batch_size: int,
):
    # cut the image into patches of size patch_size
    # format the image patches to feed the model.predict function
    predictions_dataset = build_predictions_dataset(target_image_path, patch_size)

    # build the model
    # apply saved weights to the built model
    model = load_saved_model(
        checkpoint_dir_path=checkpoint_dir_path,
        n_classes=n_classes,
        input_shape=patch_size,
        batch_size=batch_size,
    )

    # make predictions on the patches
    logger.info("\nStart to make predictions...")
    predictions = model.predict(predictions_dataset)
    classes = list(np.argmax(predictions, axis=3))
    logger.info("\nPredictions have been done.")

    # remove background predictions so it we take the max on the non background classes

    # rebuild the image with the predictions patches -> cf image_rebuilder.py
    full_predictions_tensor = rebuild_predictions(classes, target_image_path, patch_size)

    # export the predicted image -> cf save_tensor_to_jpg & plotting_tools

    return full_predictions_tensor


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
