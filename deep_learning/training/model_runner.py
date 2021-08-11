import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from dataset_utils.file_utils import timeit
from deep_learning.models.unet import build_small_unet
from dataset_utils.dataset_builder import get_dataset
from pathlib import Path


CHECKPOINT_PATH = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\checkpoints\cp.ckpt"
)
CHECKPOINT_DIR_PATH = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\checkpoints"
)
MODEL_PLOT_PATH = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files\models\model.png"
)
SAVED_PATCHES_COVERAGE_PERCENT_PATH = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files\temp_files\patches_coverage.csv"
)

BATCH_SIZE = 2
TEST_PROPORTION = 0.2
PATCH_COVERAGE_PERCENT_LIMIT = 75
N_CLASSES = 9
N_PATCHES_LIMIT = 10
INPUT_SHAPE = 256
# OPTIMIZER = "rmsprop"
OPTIMIZER = Adam(lr=1e-4)
LOSS_FUNCTION = "categorical_crossentropy"
METRICS = ["accuracy"]
EPOCHS = 5


@timeit
def main():
    # Define the model
    model = build_small_unet(N_CLASSES, INPUT_SHAPE, BATCH_SIZE)

    # Compile the model
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)

    # Init the callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH, verbose=1, save_weights_only=True
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

    # Evaluate the model
    loss, accuracy = model.evaluate(test_dataset, verbose=1)

    # Make predictions
    predictions = model.predict(test_dataset)
    classes = np.argmax(predictions, axis=3)

    breakpoint()


def make_predictions(target_image_path, patch_size: int):
    # cut the image into patches of size patch_size

    # build the model

    # apply saved weights to the built model

    # format the image patches to feed the model.predict function

    # make predictions on the patches

    # remove background predictions so it we take the max on the non background classes

    # rebuild the image with the predictions patches

    # export the predicted image

    pass


def load_saved_model():
    model = build_small_unet(N_CLASSES, INPUT_SHAPE, BATCH_SIZE)
    filepath = tf.train.latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR_PATH)
    model.load_weights(filepath=filepath)
    # the warnings logs due to load_weights are here because we don't train (compile/fit) after : they disappear if we do