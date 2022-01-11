from pathlib import Path

from loguru import logger
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

# import scikitplot as skplt

from constants import (
    PALETTE_HEXA,
    MAPPING_CLASS_NUMBER,
    IMAGE_PATH,
    MASKS_DIR_PATH,
    PREDICTIONS_DIR_PATH,
    PATCH_SIZE,
    PATCH_OVERLAP,
    N_CLASSES,
    BATCH_SIZE,
    ENCODER_KERNEL_SIZE,
    DOWNSCALE_FACTORS,
    IMAGES_DIR_PATH,
    MASK_TRUE_VALUE,
    MASK_FALSE_VALUE,
)
from dataset_utils.dataset_builder import build_predictions_dataset
from dataset_utils.file_utils import get_formatted_time
from dataset_utils.image_rebuilder import (
    rebuild_predictions_with_overlap,
)
from dataset_utils.image_utils import decode_image, get_image_name_without_extension
from dataset_utils.masks_encoder import stack_image_masks
from dataset_utils.plotting_tools import map_categorical_mask_to_3_color_channels_tensor
from deep_learning.models.unet import build_small_unet, build_small_unet_arbitrary_input


# todo : deprecate this one if make_predictions_oneshot works properly
def make_predictions(
    target_image_path: Path,
    checkpoint_dir_path: Path,
    patch_size: int,
    patch_overlap: int,
    n_classes: int,
    batch_size: int,
    encoder_kernel_size: int,
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
    :param misclassification_size: Estimated number of pixels on which the classification is wrong due to side effects between neighbors patches.

    :return: A 2D categorical tensor of size (width, height), width and height being the cropped size of the target image tensor
    """

    assert (
        patch_overlap % 2 == 0
    ), f"Patch overlap argument must be a pair number. The one specified was {patch_overlap}."

    image_tensor = decode_image(file_path=target_image_path)

    # Cut the image into patches of size patch_size
    # & format the image patches to feed the model.predict function
    main_patches_dataset, right_side_patches_dataset = build_predictions_dataset(
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
        predictions_dataset=main_patches_dataset, model=model
    )
    right_side_patch_classes_list = patches_predict(
        predictions_dataset=right_side_patches_dataset, model=model
    )

    # Rebuild the image with the predictions patches
    # output tensor of size (intput_width_size - 2 * patch_overlap, input_height_size - 2 * patch_overlap)
    # todo : check that we reach the size announced on the line above
    final_predictions_tensor = rebuild_predictions_with_overlap(
        target_image_path=target_image_path,
        main_patch_classes_list=main_patch_classes_list,
        right_side_patch_classes_list=right_side_patch_classes_list,
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
    predictions_dataset: tf.data.Dataset, model: tf.keras.Model
) -> [tf.Tensor]:
    predictions = model.predict(predictions_dataset, verbose=1)

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


def make_predictions_oneshot(
    target_image_path: Path,
    checkpoint_dir_path: Path,
    patch_overlap: int,
    n_classes: int,
    patch_size: int,
    batch_size: int,
    encoder_kernel_size: int,
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
    :param misclassification_size: Estimated number of pixels on which the classification is wrong due to side effects between neighbors patches.

    :return: A 2D categorical tensor of size (width, height), width and height being the cropped size of the target image tensor
    """

    assert (
        patch_overlap % 2 == 0
    ), f"Patch overlap argument must be a pair number. The one specified was {patch_overlap}."

    image_tensor = decode_image(file_path=target_image_path)
    image_tensor = tf.expand_dims(input=image_tensor, axis=0)

    # todo : generate patches of downsampled images

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

    # todo : fix this because error :
    #   ValueError: Dimension 1 in both shapes must be equal, but are 488 and 489.
    #   Shapes are [8,488,752] and [8,489,752]. for '{{node U-Net/concatenate_5/concat}} = ConcatV2[N=2, T=DT_FLOAT, Tidx=DT_INT32]
    #   (U-Net/conv2d_transpose_5/BiasAdd, U-Net/activation_15/Relu, U-Net/concatenate_5/concat/axis)' with input shapes:
    #   [8,488,752,32], [8,489,752,32], [] and with computed input tensors: input[2] = <3>.
    predictions = model.predict(image_tensor, verbose=1)

    return predictions


# todo : put it in the plotting_tools.py instead of reporting.py ?
def save_labels_vs_predictions_comparison_plot(
    target_image_path: Path,
    masks_dir_path: Path,
    report_dir_path: Path,
    patch_size: int,
    patch_overlap: int,
    n_classes: int,
    batch_size: int,
    encoder_kernel_size: int,
) -> None:
    """Used for training images, that have labels in the masks_dir folder !!!"""
    # Make predictions
    # predictions_tensor = make_predictions(
    #     target_image_path=target_image_path,
    #     checkpoint_dir_path=report_dir_path / "2_model_report",
    #     patch_size=patch_size,
    #     patch_overlap=patch_overlap,
    #     n_classes=n_classes,
    #     batch_size=batch_size,
    #     encoder_kernel_size=encoder_kernel_size,
    # )
    predictions_tensor = make_predictions_oneshot(
        target_image_path=target_image_path,
        checkpoint_dir_path=report_dir_path / "2_model_report",
        patch_overlap=patch_overlap,
        n_classes=n_classes,
        patch_size=patch_size,
        batch_size=batch_size,
        encoder_kernel_size=encoder_kernel_size,
    )

    # Set-up plotting settings
    image = decode_image(file_path=target_image_path).numpy()
    categorical_tensor = stack_image_masks(
        image_path=target_image_path, masks_dir_path=masks_dir_path
    )
    mapped_categorical_array = map_categorical_mask_to_3_color_channels_tensor(
        categorical_mask_tensor=categorical_tensor
    )
    mapped_predictions_array = map_categorical_mask_to_3_color_channels_tensor(
        categorical_mask_tensor=predictions_tensor
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

    # Save the plot
    # todo : save it at the root of the report predictions, (if this function is still needed)
    predictions_dir_path = (
        report_dir_path
        / "3_predictions"
        / f"{get_image_name_without_extension(target_image_path)}"
        / get_formatted_time()
    )
    labels_and_predictions_sub_dir = predictions_dir_path / "predictions_only"
    if not labels_and_predictions_sub_dir.exists():
        labels_and_predictions_sub_dir.mkdir(parents=True)
    output_path = (
        labels_and_predictions_sub_dir
        / f"{get_image_name_without_extension(target_image_path)}.png"
    )
    plt.savefig(output_path, bbox_inches="tight", dpi=300)

    logger.info(f"\nFull predictions plot successfully saved at : {output_path}")


# todo : save confusion matrix at each run
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
    :param predictions_tensor: A categorical predicitions tensor.
    :param masks_dir_path: The masks source directory path.
    :param n_classes: Total number of classes, background not included.
    :return:
    """
    # todo : remove the image_path argument from this function

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
    # note : there is a little hack with the +1 to consider the background class, and remote it just after ith [1:]
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


# def plot_confusion_matrix(
#     labels_tensor: tf.Tensor,
#     predictions_tensor: tf.Tensor,
# ) -> None:
#     skplt.metrics.plot_confusion_matrix(
#         y_true=labels_tensor,
#         y_pred=predictions_tensor,
#         figsize=(10, 10),
#         title="Confusion matrix",
#         x_tick_rotation=45,
#         cmap="Greens",
#     )
#     plt.show()

#
# def save_confusion_matrix(
#     labels_tensor: tf.Tensor,
#     predictions_tensor: tf.Tensor,
#     output_path: Path,
# ) -> None:
#     skplt.metrics.plot_confusion_matrix(
#         y_true=labels_tensor,
#         y_pred=predictions_tensor,
#         figsize=(10, 10),
#         title="Confusion matrix",
#         x_tick_rotation=45,
#         cmap="Greens",
#     )
#     plt.savefig(output_path)


# ------
# DEBUG
# make_predictions(IMAGE_PATH, CHECKPOINT_DIR_PATH, PATCH_SIZE, PATCH_OVERLAP, N_CLASSES, BATCH_SIZE, ENCODER_KERNEL_SIZE, DOWNSCALE_FACTORS)
# save_full_plot_predictions(IMAGE_PATH, MASKS_DIR_PATH, OUTPUT_DIR_PATH, CHECKPOINT_DIR_PATH, PATCH_SIZE, PATCH_OVERLAP, N_CLASSES, BATCH_SIZE, ENCODER_KERNEL_SIZE, DOWNSCALE_FACTORS)
# save_predictions_plot_only(IMAGE_PATH, PREDICTIONS_DIR_PATH, CHECKPOINT_DIR_PATH, PATCH_SIZE, PATCH_OVERLAP, N_CLASSES, BATCH_SIZE, ENCODER_KERNEL_SIZE, DOWNSCALE_FACTORS)

#
# image_path = IMAGES_DIR_PATH / "_DSC0246/_DSC0246.jpg"
# predictions_tensor = make_predictions(
#     image_path,
#     CHECKPOINT_DIR_PATH,
#     PATCH_SIZE,
#     PATCH_OVERLAP,
#     N_CLASSES,
#     BATCH_SIZE,
#     ENCODER_KERNEL_SIZE,
#     downscale_factors=(1, 1, 1),
# )
#
# confusion_matrix, labels_tensor, predictions_tensor = get_confusion_matrix(
#     image_path=image_path,
#     predictions_tensor=predictions_tensor,
#     masks_dir_path=MASKS_DIR_PATH,
#     n_classes=N_CLASSES,
#     patch_overlap=PATCH_OVERLAP,
# )
#
# plot_confusion_matrix(
#     labels_tensor=labels_tensor, predictions_tensor=predictions_tensor
# )
# save_confusion_matrix(
#     labels_tensor=labels_tensor,
#     predictions_tensor=predictions_tensor,
#     output_path=Path(""),
# )

# todo : generate patches of downsampled images
