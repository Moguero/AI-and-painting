from pathlib import Path
from loguru import logger
import numpy as np

from dataset_utils.dataset_builder import build_predictions_dataset
from dataset_utils.image_rebuilder import rebuild_predictions
from deep_learning.training.model_runner import load_saved_model


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
