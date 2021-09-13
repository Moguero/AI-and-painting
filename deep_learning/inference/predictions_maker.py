from pathlib import Path

from loguru import logger
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from constants import PALETTE_HEXA, MAPPING_CLASS_NUMBER, IMAGE_PATH, MASKS_DIR_PATH, OUTPUT_DIR_PATH, \
    CHECKPOINT_DIR_PATH, PATCH_SIZE, PATCH_OVERLAP, N_CLASSES, BATCH_SIZE, ENCODER_KERNEL_SIZE
from dataset_utils.dataset_builder import build_predictions_dataset
from dataset_utils.file_utils import timeit
from dataset_utils.image_rebuilder import (
    rebuild_predictions_with_overlap,
)
from dataset_utils.image_utils import decode_image, get_image_name_without_extension
from dataset_utils.masks_encoder import stack_image_masks
from dataset_utils.plotting_tools import map_categorical_mask_to_3_color_channels_tensor
from deep_learning.training.model_runner import load_saved_model


def make_predictions(
    target_image_path,
    checkpoint_dir_path: Path,
    patch_size: int,
    patch_overlap: int,
    n_classes: int,
    batch_size: int,
    encoder_kernel_size
):
    """
    Make predictions on the target image specified with its path.

    :param target_image_path: Image to make predictions on.
    :param checkpoint_dir_path: Path of the already trained model.
    :param patch_size: Size of the patches on which the model was trained. This is also the size of the predictions patches.
    :param patch_overlap: Number of pixels on which neighbors patches intersect each other.
    :param n_classes: Number of classes to map, background excluded.
    :param batch_size: Batch size that was used for the model which is loaded.

    :return: A 2D categorical tensor of size (width, height), width and height being the cropped size of the target image tensor.
    """
    # downscale the image if necessary
    ...
    # todo : downscale the image


    # cut the image into patches of size patch_size
    # & format the image patches to feed the model.predict function
    predictions_dataset = build_predictions_dataset(
        target_image_path, patch_size, patch_overlap
    )

    # build the model
    # & apply saved weights to the built model
    model = load_saved_model(
        checkpoint_dir_path=checkpoint_dir_path,
        n_classes=n_classes,
        input_shape=patch_size,
        batch_size=batch_size,
        encoder_kernel_size=encoder_kernel_size
    )

    # make predictions on the patches
    # & remove background predictions so it we take the max on the non background classes
    logger.info("\nStart to make predictions...")
    predictions = model.predict(predictions_dataset, verbose=1)
    # predicitons : array of shape (n_patches, patch_size, patch_size, n_classes)

    classes = list(np.argmax(predictions[:, :, :, 1:], axis=3))
    classes = [
        tf.expand_dims(
            tf.constant(
                np.add(patch_classes, np.ones(patch_classes.shape, dtype=np.int32)),
                dtype=tf.int32,
            ),
            axis=2,
        )
        for patch_classes in classes
    ]
    # classes list of size n_patches of tensor with size (patch_size, patch_size, 1)
    logger.info("\nPredictions have been done.")

    # rebuild the image with the predictions patches
    full_predictions_tensor = rebuild_predictions_with_overlap(
        classes, target_image_path, patch_size, patch_overlap
    )

    full_predictions_tensor = tf.squeeze(full_predictions_tensor)
    return full_predictions_tensor


@timeit
def save_full_plot_predictions(
    target_image_path: Path,
    masks_dir: Path,
    output_dir_path: Path,
    checkpoint_dir_path: Path,
    patch_size: int,
    patch_overlap: int,
    n_classes: int,
    batch_size: int,
    encoder_kernel_size: int
) -> None:
    predictions_tensor = make_predictions(
        target_image_path, checkpoint_dir_path, patch_size, patch_overlap, n_classes, batch_size, encoder_kernel_size
    )
    image = decode_image(target_image_path).numpy()
    categorical_tensor = stack_image_masks(target_image_path, masks_dir)
    mapped_categorical_array = map_categorical_mask_to_3_color_channels_tensor(
        categorical_tensor
    )
    mapped_predictions_array = map_categorical_mask_to_3_color_channels_tensor(
        predictions_tensor
    )
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle(get_image_name_without_extension(target_image_path))
    ax1.set_title("Original image")
    ax2.set_title("Predictions")
    ax3.set_title("Labels (ground truth)")
    ax1.imshow(image)
    ax2.imshow(mapped_predictions_array)
    ax3.imshow(mapped_categorical_array)
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")

    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size("x-small")
    handles = [
        matplotlib.patches.Patch(
            color=PALETTE_HEXA[MAPPING_CLASS_NUMBER[class_name]], label=class_name
        )
        for class_name in MAPPING_CLASS_NUMBER.keys()
    ]
    ax3.legend(handles=handles, bbox_to_anchor=(1.4, 1), loc="upper left", prop=fontP)

    image_sub_dir = output_dir_path / f"{get_image_name_without_extension(target_image_path)}"
    if not image_sub_dir.exists():
        image_sub_dir.mkdir()
    output_path = (
        image_sub_dir
        / f"{get_image_name_without_extension(target_image_path)}_predictions__model_{checkpoint_dir_path.parts[-1]}__overlap_{patch_overlap}.png"
    )
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    logger.info(f"\nFull predictions plot successfully saved at : {output_path}")
    # todo : add unique identification to same image/same model file (like random id) to compare with different values of misclassification_size


def save_predictions_plot_only(
    target_image_path: Path,
    output_dir_path: Path,
    checkpoint_dir_path: Path,
    patch_size: int,
    patch_overlap: int,
    n_classes: int,
    batch_size: int,
    encoder_kernel_size: int
) -> None:
    predictions_tensor = make_predictions(
        target_image_path, checkpoint_dir_path, patch_size, patch_overlap, n_classes, batch_size, encoder_kernel_size
    )
    mapped_predictions_array = map_categorical_mask_to_3_color_channels_tensor(
        predictions_tensor
    )
    image_sub_dir = output_dir_path / f"{get_image_name_without_extension(target_image_path)}" / "predictions_only"
    if not image_sub_dir.exists():
        image_sub_dir.mkdir()
    output_path = (
        image_sub_dir
        / f"{get_image_name_without_extension(target_image_path)}_predictions__model_{checkpoint_dir_path.parts[-1]}__overlap_{patch_overlap}.png"
    )
    tf.keras.preprocessing.image.save_img(output_path, mapped_predictions_array)
    logger.info(f"\nFull predictions plot successfully saved at : {output_path}")


# ------
# DEBUG
# predictions = make_predictions(IMAGE_PATH, CHECKPOINT_DIR_PATH, PATCH_SIZE, PATCH_OVERLAP, N_CLASSES, BATCH_SIZE)
# save_full_plot_predictions(IMAGE_PATH, MASKS_DIR_PATH, OUTPUT_DIR_PATH, CHECKPOINT_DIR_PATH, PATCH_SIZE, PATCH_OVERLAP, N_CLASSES, BATCH_SIZE, ENCODER_KERNEL_SIZE)
# save_predictions_plot_only(IMAGE_PATH, OUTPUT_DIR_PATH, CHECKPOINT_DIR_PATH, PATCH_SIZE, PATCH_OVERLAP, N_CLASSES, BATCH_SIZE, ENCODER_KERNEL_SIZE)

# todo : create a windows and linux environment_win.yml file