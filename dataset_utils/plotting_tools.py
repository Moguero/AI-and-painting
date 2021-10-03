import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np

from constants import MAPPING_CLASS_NUMBER, PALETTE_RGB_NORMALIZED, PALETTE_RGB, PALETTE_HEXA
from dataset_utils.file_utils import timeit
from dataset_utils.image_utils import decode_image, get_image_name_without_extension
from dataset_utils.image_utils import get_image_masks_paths, get_mask_class
from dataset_utils.masks_encoder import stack_image_masks
from pathlib import Path


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


def plot_image_from_array(array: np.ndarray) -> None:
    plt.imshow(array)
    plt.axis("off")
    plt.show()


@timeit
def map_categorical_mask_to_3_color_channels_tensor(
    categorical_mask_tensor: tf.Tensor,
) -> np.ndarray:
    """
    Turn a 2D tensor into a 3D array by converting its categorical values in its RGB correspondant values.

    :param categorical_mask_tensor: A 2D categorical tensor of size (width_size, height_size)
    :return: A 3D array of size (width_size, height_size, 3)
    """
    categorical_mask_array = categorical_mask_tensor.numpy()
    vectorize_function = np.vectorize(lambda x: PALETTE_RGB[x])
    three_channels_array = np.stack(vectorize_function(categorical_mask_array), axis=2)
    return three_channels_array


@timeit
def full_plot_image(
    image_path: Path, masks_dir: Path, predictions_tensor: tf.Tensor, all_patch_masks_overlap_indices_path: Path
) -> None:
    image = decode_image(image_path).numpy()
    categorical_tensor = stack_image_masks(image_path, masks_dir, all_patch_masks_overlap_indices_path)
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
        matplotlib.patches.Patch(color=PALETTE_HEXA[MAPPING_CLASS_NUMBER[class_name]], label=class_name)
        for class_name in MAPPING_CLASS_NUMBER.keys()
    ]
    ax3.legend(handles=handles, bbox_to_anchor=(1.4, 1), loc="upper left", prop=fontP)

    plt.show()


# save_full_plot_image(IMAGE_PATH, MASKS_DIR, predictions, OUTPUT_PATH)



@timeit
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

# save_plot_predictions_only(IMAGE_PATH, predictions, DATA_DIR_ROOT / "predictions/test2.jpg")


# todo : develop this plot
# {'loss': [2.4902284145355225, 2.272948980331421, 2.180922746658325, 2.123626708984375, 2.0806949138641357, 2.0426666736602783, 2.0098025798797607, 1.9834508895874023, 1.954201102256775, 1.9568171501159668], 'accuracy': [0.12945209443569183, 0.20466716587543488, 0.24759912490844727, 0.2723337709903717, 0.2895956039428711, 0.3004612624645233, 0.3144834637641907, 0.32735636830329895, 0.34158948063850403, 0.3489217460155487]}
def plot_training_history(history: keras.callbacks.History) -> None:
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(["train"], loc='upper left')
    plt.grid(True)
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train'], loc='upper left')
    plt.grid(True)
    plt.show()
