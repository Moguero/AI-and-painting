import scipy
import numpy as np
import tensorflow as tf
from loguru import logger
from pathlib import Path

from dataset_utils.dataset_builder import build_predictions_dataset
from dataset_utils.image_rebuilder import (
    rebuild_predictions_with_overlap,
)
from dataset_utils.image_utils import decode_image, get_image_name_without_extension
from dataset_utils.masks_encoder import stack_image_masks
from deep_learning.unet import build_small_unet
from constants import MAPPING_CLASS_NUMBER


def make_predictions(
    target_image_path: Path,
    checkpoint_dir_path: Path,
    patch_size: int,
    patch_overlap: int,
    n_classes: int,
    batch_size: int,
    encoder_kernel_size: int,
    correlate_predictions_bool: bool,
    correlation_filter: np.ndarray,
    misclassification_size: int = 5,
) -> tf.Tensor:
    """
    Make predictions on the target image specified with its path.

    :param encoder_kernel_size: Size of the kernel encoder.
    :param target_image_path: Image to make predictions on.
    :param checkpoint_dir_path: Path of the already trained model.
    :param patch_size: Size of the patches on which the model was trained. This is also the size of the predictions patches.
    :param patch_overlap: Number of pixels on which neighbors patches intersect each other.
    :param n_classes: Number of classes to map, background excluded.
    :param batch_size: Batch size that was used for the model which is loaded.
    :param correlate_predictions_bool: Whether or not to apply correlation on the predictions, i.e. make a local weighted mean on each class probability.
    :param correlation_filter: The filter to use for correlation.
    :param misclassification_size: Estimated number of pixels on which the classification is wrong due to side effects between neighbors patches.

    :return: A 2D categorical tensor of size (width, height), width and height being the cropped size of the target image tensor
    """

    assert (
        patch_overlap % 2 == 0
    ), f"Patch overlap argument must be a pair number. The one specified was {patch_overlap}."

    image_tensor = decode_image(file_path=target_image_path)

    # Cut the image into patches of size patch_size
    # & format the image patches to feed the model.predict function
    (
        main_patches_dataset,
        right_side_patches_dataset,
        down_side_patches_dataset,
    ) = build_predictions_dataset(
        target_image_tensor=image_tensor,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )

    # Build the model
    # & apply saved weights to the built model
    model = load_saved_model(
        checkpoint_dir_path=checkpoint_dir_path,
        n_classes=n_classes,
        input_shape=patch_size,
        batch_size=batch_size,
        encoder_kernel_size=encoder_kernel_size,
    )

    # Make predictions on the patches
    # output : array of shape (n_patches, patch_size, patch_size, n_classes)
    main_patch_classes_list = patches_predict(
        predictions_dataset=main_patches_dataset,
        model=model,
        n_classes=n_classes,
        correlate_predictions_bool=correlate_predictions_bool,
        correlation_filter=correlation_filter,
    )
    right_side_patch_classes_list = patches_predict(
        predictions_dataset=right_side_patches_dataset,
        model=model,
        n_classes=n_classes,
        correlate_predictions_bool=correlate_predictions_bool,
        correlation_filter=correlation_filter,
    )
    down_side_patch_classes_list = patches_predict(
        predictions_dataset=down_side_patches_dataset,
        model=model,
        n_classes=n_classes,
        correlate_predictions_bool=correlate_predictions_bool,
        correlation_filter=correlation_filter,
    )

    # Rebuild the image with the predictions patches
    # output tensor of size (intput_width_size - 2 * patch_overlap, input_height_size - 2 * patch_overlap)
    final_predictions_tensor = rebuild_predictions_with_overlap(
        target_image_path=target_image_path,
        main_patch_classes_list=main_patch_classes_list,
        right_side_patch_classes_list=right_side_patch_classes_list,
        down_side_patch_classes_list=down_side_patch_classes_list,
        image_tensor=image_tensor,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        misclassification_size=misclassification_size,
    )

    logger.info(
        f"\nPredictions on {get_image_name_without_extension(target_image_path)} have been done."
    )

    return final_predictions_tensor


def patches_predict(
    predictions_dataset: tf.data.Dataset,
    model: tf.keras.Model,
    n_classes: int,
    correlate_predictions_bool: bool,
    correlation_filter: np.ndarray,
) -> [tf.Tensor]:
    predictions: np.ndarray = model.predict(x=predictions_dataset, verbose=1)

    if correlate_predictions_bool:
        predictions = correlate_predictions(
            predictions_array=predictions,
            correlation_filter=correlation_filter,
            n_classes=n_classes,
        )

    # Remove background predictions so it takes the max on the non background classes
    # Note : the argmax function shift the classes numbers of -1, that is why we add one just after
    patch_classes_list = list(np.argmax(predictions[:, :, :, 1:], axis=3))
    patch_classes_list = [
        tf.constant(
            np.add(patch_classes, np.ones(patch_classes.shape, dtype=np.int32)),
            dtype=tf.int32,
        )
        for patch_classes in patch_classes_list
    ]
    # Tensors classes list is of size n_patches of tensor with size (patch_size, patch_size)

    return patch_classes_list


def correlate_predictions(
    predictions_array: np.ndarray, correlation_filter: np.ndarray, n_classes: int
) -> tf.Tensor:
    """Smooth the predictions by applying a gaussian filter to the predictions probabilities.
    It computes a local weighted average on each class probabilities."""
    assert (
        predictions_array.shape[-1] == n_classes + 1
    ), f"Predictions tensor has shape {predictions_array.shape} (last axis dim {predictions_array.shape[-1]}), last axis should have dim {n_classes + 1}"

    n_patches = predictions_array.shape[0]
    correlated_patches_predictions_list = list()
    for patch_tensor_idx in range(n_patches):
        correlated_classes_means_list = list()
        for class_idx in range(n_classes + 1):
            # compute a local weighted average
            correlated_class_means = (
                scipy.ndimage.correlate(
                    input=predictions_array[patch_tensor_idx, :, :, class_idx],
                    weights=correlation_filter,
                )
                / int(tf.reduce_sum(correlation_filter))
            )
            # remark : no need to normalize the weighted probabilities because only an argmax is done after
            correlated_classes_means_list.append(correlated_class_means)
        patch_correlated_predictions = tf.stack(
            values=correlated_classes_means_list, axis=-1
        )
        correlated_patches_predictions_list.append(patch_correlated_predictions)
    correlated_predictions = tf.stack(
        values=correlated_patches_predictions_list, axis=0
    )
    return correlated_predictions


def make_predictions_oneshot(
    target_image_path: Path,
    checkpoint_dir_path: Path,
    patch_overlap: int,
    n_classes: int,
    patch_size: int,
    batch_size: int,
    encoder_kernel_size: int,
) -> tf.Tensor:
    """,
    Make predictions on the target image specified with its path.

    :param encoder_kernel_size: Size of the kernel encoder.
    :param target_image_path: Image to make predictions on.
    :param checkpoint_dir_path: Path of the already trained model.
    :param patch_size: Size of the patches on which the model was trained. This is also the size of the predictions patches.
    :param patch_overlap: Number of pixels on which neighbors patches intersect each other.
    :param n_classes: Number of classes to map, background excluded.
    :param batch_size: Batch size that was used for the model which is loaded.
    :param misclassification_size: Estimated number of pixels on which the classification is wrong due to side effects between neighbors patches.

    :return: A 2D categorical tensor of size (width, height), width and height being the cropped size of the target image tensor
    """

    assert (
        patch_overlap % 2 == 0
    ), f"Patch overlap argument must be a pair number. The one specified was {patch_overlap}."

    image_tensor = decode_image(file_path=target_image_path)
    image_tensor = tf.expand_dims(input=image_tensor, axis=0)

    # Build the model
    # & apply saved weights to the built model
    model = load_saved_model(
        checkpoint_dir_path=checkpoint_dir_path,
        n_classes=n_classes,
        input_shape=patch_size,
        batch_size=batch_size,
        encoder_kernel_size=encoder_kernel_size,
    )

    # Make predictions on the patches
    # predicitons : array of shape (n_patches, patch_size, patch_size, n_classes)
    predictions = model.predict(image_tensor, verbose=1)

    return predictions


def get_confusion_matrix(
    image_path: Path,
    predictions_tensor: tf.Tensor,
    masks_dir_path: Path,
    n_classes: int,
    patch_overlap: int,
) -> (tf.Tensor, tf.Tensor, tf.Tensor):
    """
    Compute the confusion matrix for the predictions input tensor regarding its labels.

    :param image_path: The path of the image on which the predictions are made
    :param predictions_tensor: A categorical predictions tensor.
    :param masks_dir_path: The masks source directory path.
    :param n_classes: Total number of classes, background not included.
    """

    assert (
        patch_overlap % 2 == 0
    ), f"Patch overlap argument must be a pair number. The one specified was {patch_overlap}."

    # load the labels
    labels_tensor = stack_image_masks(
        image_path=image_path, masks_dir_path=masks_dir_path
    )

    assert (
        labels_tensor.shape == decode_image(file_path=image_path)[:, :, 0].shape
    ), f"Labels tensor shape is {labels_tensor.shape} while it should be {decode_image(file_path=image_path)[:, :, 0].shape}"

    labels_tensor = labels_tensor[
        patch_overlap // 2 : predictions_tensor.shape[0] + patch_overlap // 2,
        patch_overlap // 2 : predictions_tensor.shape[1] + patch_overlap // 2,
    ]

    # flatten both labels and predictions
    predictions_tensor = tf.reshape(predictions_tensor, [-1])
    labels_tensor = tf.reshape(labels_tensor, [-1])

    # drop parts where the labels are equal to "background" class in both tensors
    predictions_tensor = tf.boolean_mask(
        tensor=predictions_tensor,
        mask=tf.not_equal(labels_tensor, MAPPING_CLASS_NUMBER["background"]),
    )
    labels_tensor = tf.boolean_mask(
        tensor=labels_tensor,
        mask=tf.not_equal(labels_tensor, MAPPING_CLASS_NUMBER["background"]),
    )

    assert predictions_tensor.shape == labels_tensor.shape

    # get the confusion matrix
    # note : there is a little hack with the +1 to consider the background class, and remove it just after ith [1:]
    confusion_matrix = tf.math.confusion_matrix(
        labels=labels_tensor, predictions=predictions_tensor, num_classes=n_classes + 1
    )[1:, 1:]

    return confusion_matrix, labels_tensor, predictions_tensor


def load_saved_model(
    checkpoint_dir_path: Path,
    n_classes: int,
    input_shape: int,
    batch_size: int,
    encoder_kernel_size: int,
):
    logger.info("\nLoading the model...")
    # model = build_small_unet(n_classes, patch_size, batch_size, encoder_kernel_size)
    model = build_small_unet(
        n_classes=n_classes,
        input_shape=input_shape,
        batch_size=batch_size,
        encoder_kernel_size=encoder_kernel_size,
    )
    filepath = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir_path)
    model.load_weights(filepath=filepath)
    # the warnings logs due to load_weights are here because we don't train (compile/fit) after : they disappear if we do
    logger.info("\nModel loaded successfully.")
    return model


# ------
# DEBUG

# make_predictions(IMAGE_PATH, CHECKPOINT_DIR_PATH, PATCH_SIZE, PATCH_OVERLAP, N_CLASSES, BATCH_SIZE, ENCODER_KERNEL_SIZE, DOWNSCALE_FACTORS)
# test_image_path = Path(r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files\test_images\downscaled_images\max\downscaled_max__DSC0245.jpg")
# predictions_tensor = make_predictions(
#     target_image_path=test_image_path,
#     checkpoint_dir_path=CHECKPOINT_DIR_PATH,
#     patch_size=PATCH_SIZE,
#     patch_overlap=PATCH_OVERLAP,
#     n_classes=N_CLASSES,
#     batch_size=BATCH_SIZE,
#     encoder_kernel_size=ENCODER_KERNEL_SIZE,
# )
#
# confusion_matrix, labels_tensor, predictions_tensor = get_confusion_matrix(
#     image_path=TEST_IMAGE_PATH,
#     predictions_tensor=predictions_tensor,
#     masks_dir_path=MASKS_DIR_PATH,
#     n_classes=N_CLASSES,
#     patch_overlap=PATCH_OVERLAP,
# )
