import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from pathlib import Path

from dataset_builder.masks_encoder import stack_image_masks
from utils.image_utils import (
    get_image_masks_paths,
    get_mask_class,
    decode_image,
    get_image_name_without_extension,
)
from constants import (
    MAPPING_CLASS_NUMBER,
    PALETTE_RGB_NORMALIZED,
    PALETTE_RGB,
    PALETTE_HEXA,
)


def plot_mask_with_color(image_path: Path, mask_path: Path) -> None:
    """Plot a mask with transparent color and with the associated picture in the background."""
    image_name = get_image_name_without_extension(image_path)
    assert (
        image_name == mask_path.parts[-3]
    ), f"The mask {mask_path} doesn't match with the images {image_path}"
    image = decode_image(image_path).numpy()
    mask_tensor = tf.cast(decode_image(mask_path), float)
    color = [1, 0, 0, 0.5]  # RED
    color_tensor = tf.constant([[color] * mask_tensor.shape[1]] * mask_tensor.shape[0])
    mask = tf.multiply(color_tensor, mask_tensor).numpy().astype(int)
    plt.imshow(image)
    plt.imshow(mask)
    plt.axis("off")
    plt.show()


def plot_colored_masks_with_background_image(image_path: Path, masks_dir: Path) -> None:
    """Plot all the masks of an images with a random color, with the images in the background."""
    image = decode_image(image_path).numpy()
    plt.imshow(image)
    image_masks_paths = get_image_masks_paths(image_path, masks_dir)
    for mask_path in image_masks_paths:
        mask_tensor = tf.cast(decode_image(mask_path), float)
        mask_class = get_mask_class(mask_path)
        class_number = MAPPING_CLASS_NUMBER[mask_class]
        mask_color = list(PALETTE_RGB_NORMALIZED[class_number]) + [0.5]
        color_tensor = tf.constant(
            [[mask_color] * mask_tensor.shape[1]] * mask_tensor.shape[0]
        )
        mask = tf.multiply(color_tensor, mask_tensor).numpy().astype(int)
        plt.imshow(mask)
    plt.axis("off")
    plt.show()


def save_colored_masks_with_background_image(
    image_path: Path, masks_dir: Path, output_path: Path
) -> None:
    """Plot all the masks of an images with a random color, with the images in the background."""
    image = decode_image(image_path).numpy()
    plt.imshow(image)
    image_masks_paths = get_image_masks_paths(image_path, masks_dir)
    for mask_path in image_masks_paths:
        mask_tensor = tf.cast(decode_image(mask_path), float)
        mask_class = get_mask_class(mask_path)
        class_number = MAPPING_CLASS_NUMBER[mask_class]
        mask_color = list(PALETTE_RGB_NORMALIZED[class_number]) + [0.5]
        color_tensor = tf.constant(
            [[mask_color] * mask_tensor.shape[1]] * mask_tensor.shape[0]
        )
        mask = tf.multiply(color_tensor, mask_tensor).numpy().astype(int)
        plt.imshow(mask)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight")


def plot_image_from_tensor(tensor: tf.Tensor) -> None:
    image = tensor.numpy()
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def save_image_from_tensor(
    tensor: tf.Tensor, output_path: Path, title: str = ""
) -> None:
    image = tensor.numpy()
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)


def plot_image_from_array(array: np.ndarray) -> None:
    plt.imshow(array)
    plt.axis("off")
    plt.show()


def map_categorical_mask_to_3_color_channels_tensor(
    categorical_mask_tensor: tf.Tensor,
) -> np.ndarray:
    """
    Turn a 2D tensor into a 3D array by converting its categorical values in its RGB correspondent values.

    :param categorical_mask_tensor: A 2D categorical tensor of size (width_size, height_size)
    :return: A 3D array of size (width_size, height_size, 3)
    """
    categorical_mask_array = categorical_mask_tensor.numpy()
    vectorize_function = np.vectorize(lambda x: PALETTE_RGB[x])
    three_channels_array = np.stack(vectorize_function(categorical_mask_array), axis=2)
    return three_channels_array


def turn_2d_tensor_to_3d_tensor(
    tensor_2d: tf.Tensor,
) -> np.ndarray:
    """
    Turn a 2D tensor into a 3D array by tripling the same mask layer.

    :param tensor_2d: A 2D categorical tensor of size (width_size, height_size)
    :return: A 3D array of size (width_size, height_size, 3)
    """
    array_2d = tensor_2d.numpy()
    vectorize_function = np.vectorize(lambda x: (x, x, x))
    array_3d = np.stack(vectorize_function(array_2d), axis=2)
    return array_3d


def full_plot_image(
    image_path: Path,
    masks_dir: Path,
    predictions_tensor: tf.Tensor,
) -> None:
    image = decode_image(image_path).numpy()
    categorical_tensor = stack_image_masks(image_path, masks_dir)
    mapped_categorical_array = map_categorical_mask_to_3_color_channels_tensor(
        categorical_tensor
    )
    mapped_predictions_array = map_categorical_mask_to_3_color_channels_tensor(
        predictions_tensor
    )
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle(get_image_name_without_extension(image_path))
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

    plt.show()


def save_plot_predictions_only(
    image_path: Path, predictions_tensor: tf.Tensor, output_path: Path
) -> None:
    plt.cla()
    mapped_predictions_array = map_categorical_mask_to_3_color_channels_tensor(
        predictions_tensor
    )
    plt.title(f"Predictions : {get_image_name_without_extension(image_path)}")
    plt.imshow(mapped_predictions_array)
    plt.axis("off")

    plt.savefig(output_path, bbox_inches="tight", dpi=300)


def plot_training_history(history: keras.callbacks.History) -> None:
    # summarize history for accuracy
    plt.plot(history.history["accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train"], loc="upper left")
    plt.grid(True)
    plt.show()
    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train"], loc="upper left")
    plt.grid(True)
    plt.show()


def save_patch_composition_plot(
    patch_composition_stats_dict: {str: float},
    output_path: Path,
    palette_hexa: {int: str},
):
    fig, ax = plt.subplots()

    classes = list(patch_composition_stats_dict.keys())
    means = [
        round((percent * 100), 1)
        for percent in list(patch_composition_stats_dict.values())
    ]
    colors = [color for color in palette_hexa.values()]

    bars = ax.barh(classes, means, color=colors)
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width * 0.95,
            bar.get_y() + bar.get_height() / 2,
            "%d" % int(width),
            ha="right",
            va="center",
        )

    ax.set_xlabel("Class proportion (%)")
    ax.set_title("Patches composition")

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
