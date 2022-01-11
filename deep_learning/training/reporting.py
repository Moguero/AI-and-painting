from pathlib import Path

import matplotlib
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf

from constants import PALETTE_HEXA, MAPPING_CLASS_NUMBER, MASK_TRUE_VALUE, MASK_FALSE_VALUE, TEST_IMAGES_PATHS_LIST
from dataset_utils.file_utils import get_formatted_time
from dataset_utils.image_utils import decode_image, get_image_name_without_extension
from dataset_utils.plotting_tools import (
    save_patch_composition_mean_plot,
    map_categorical_mask_to_3_color_channels_tensor, turn_2d_tensor_to_3d_tensor,
)
from deep_learning.inference.predictions_maker import make_predictions, make_predictions_oneshot


def build_training_run_report(
    report_dir_path: Path,
    model: keras.Model,
    model_config: dict,
    patches_composition_stats: pd.DataFrame,
    palette_hexa: {int: str},
    note: str,
) -> None:
    report_subdirs_paths_dict = {
        "data_report": report_dir_path / "1_data_report",
        "model_report": report_dir_path / "2_model_report",
        "predictions": report_dir_path / "3_predictions",
    }
    for subdir in report_subdirs_paths_dict.values():
        if not subdir.exists():
            subdir.mkdir(parents=True)

    # 0. Add note to the report if there is one
    if note != "":
        note_filename = report_dir_path / "run_note.txt"
        with open(note_filename, "w") as file:
            # Pass the file handle in as a lambda function to make it callable
            file.write(note)

    # 1. data report

    # Plot patches labels composition
    patches_composition_mean_stats_dict = patches_composition_stats.loc[
        "mean"
    ].to_dict()
    patch_composition_mean_plot_output_path = (
        report_subdirs_paths_dict["data_report"] / "patches_composition.png"
    )
    save_patch_composition_mean_plot(
        patch_composition_stats_dict=patches_composition_mean_stats_dict,
        output_path=patch_composition_mean_plot_output_path,
        palette_hexa=palette_hexa,
    )

    # Save the figures with which the plot was made
    patch_composition_mean_output_path = (
        report_subdirs_paths_dict["data_report"] / "patches_composition.txt"
    )
    with open(patch_composition_mean_output_path, "w") as file:
        for key, value in patches_composition_mean_stats_dict.items():
            file.write(f"{key}: {value},\n")

    # Save patches used for training
    # todo : plot minimized patches used (array of patches)

    # 2. model report

    # Export the model architecture
    model_architecture_filename = (
        report_subdirs_paths_dict["model_report"] / "model_architecture.txt"
    )
    with open(model_architecture_filename, "w") as file:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: file.write(x + "\n"))

    # Export the model config (hyperparameters)

    model_config_filename = (
        report_subdirs_paths_dict["model_report"] / "model_config.txt"
    )
    with open(model_config_filename, "w") as file:
        # Pass the file handle in as a lambda function to make it callable
        for key, value in model_config.items():
            file.write(f"{str(key)}: {str(value)} \n")
    return


# todo : rethink the use of this function : is it really useful ?
def init_report_paths(report_root_dir_path: Path) -> {str: Path}:
    _report_dir_path = report_root_dir_path / f"report_{get_formatted_time()}"
    _data_report = _report_dir_path / "1_data_report"
    _model_report = _report_dir_path / "2_model_report"
    _predictions_report = _report_dir_path / "3_predictions_report"
    _checkpoint_path = _model_report / "model_checkpoint"

    report_paths_dict = dict()
    report_paths_dict["report_dir_path"] = _report_dir_path
    report_paths_dict["data_report"] = _data_report
    report_paths_dict["model_report"] = _model_report
    report_paths_dict["predictions_report"] = _predictions_report
    report_paths_dict["checkpoint_path"] = _checkpoint_path

    return report_paths_dict


def build_predict_run_report(
    test_images_paths_list: [Path],
    report_dir_path: Path,
    patch_size: int,
    patch_overlap: int,
    n_classes: int,
    batch_size: int,
    encoder_kernel_size: int,
) -> None:
    predictions_report_root_path = (
        report_dir_path / "3_predictions" / get_formatted_time()
    )
    predictions_report_root_path.mkdir(parents=True)

    for test_image_path in test_images_paths_list:
        # Make predictions
        predictions_tensor = make_predictions(
            target_image_path=test_image_path,
            checkpoint_dir_path=report_dir_path / "2_model_report",
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            n_classes=n_classes,
            batch_size=batch_size,
            encoder_kernel_size=encoder_kernel_size,
        )

        save_test_images_vs_predictions_plot(
            target_image_path=test_image_path,
            predictions_tensor=predictions_tensor,
            predictions_report_root_path=predictions_report_root_path,
        )

        save_predictions_only_plot(
            target_image_path=test_image_path,
            predictions_tensor=predictions_tensor,
            predictions_report_root_path=predictions_report_root_path,
        )

        save_binary_predictions_plot(
            target_image_path=test_image_path,
            predictions_tensor=predictions_tensor,
            predictions_report_root_path=predictions_report_root_path,
        )

        save_predictions_config(
            predictions_report_root_path=predictions_report_root_path,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            batch_size=batch_size,
            encoder_kernel_size=encoder_kernel_size,
        )


def save_test_images_vs_predictions_plot(
    target_image_path: Path,
    predictions_tensor: tf.Tensor,
    predictions_report_root_path: Path,
) -> None:
    # Set-up plotting settings
    image = decode_image(file_path=target_image_path).numpy()
    mapped_predictions_array = map_categorical_mask_to_3_color_channels_tensor(
        categorical_mask_tensor=predictions_tensor
    )

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(get_image_name_without_extension(image_path=target_image_path))
    ax1.set_title("Original image")
    ax2.set_title("Predictions")
    ax1.imshow(image)
    ax2.imshow(mapped_predictions_array)
    ax1.axis("off")
    ax2.axis("off")

    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size("x-small")
    handles = [
        matplotlib.patches.Patch(
            color=PALETTE_HEXA[MAPPING_CLASS_NUMBER[class_name]], label=class_name
        )
        for class_name in MAPPING_CLASS_NUMBER.keys()
    ]
    ax2.legend(handles=handles, bbox_to_anchor=(1.4, 1), loc="upper center", prop=fontP)

    # Save the plot
    images_and_predictions_dir_path = (
        predictions_report_root_path / "images_and_predictions"
    )
    if not images_and_predictions_dir_path.exists():
        images_and_predictions_dir_path.mkdir()

    output_path = (
        images_and_predictions_dir_path
        / f"image_vs_predictions__{get_image_name_without_extension(target_image_path)}.png"
    )
    plt.savefig(output_path, bbox_inches="tight", dpi=300)

    logger.info(
        f"\nTest image vs predictions plot successfully saved at : {output_path}"
    )


def save_predictions_only_plot(
    target_image_path: Path,
    predictions_tensor: tf.Tensor,
    predictions_report_root_path: Path,
) -> None:
    mapped_predictions_array = map_categorical_mask_to_3_color_channels_tensor(
        categorical_mask_tensor=predictions_tensor
    )
    predictions_only_subdir_path = predictions_report_root_path / "predictions_only"
    if not predictions_only_subdir_path.exists():
        predictions_only_subdir_path.mkdir()

    output_path = (
        predictions_only_subdir_path
        / f"predictions_only__{get_image_name_without_extension(target_image_path)}.png"
    )
    tf.keras.preprocessing.image.save_img(output_path, mapped_predictions_array)
    logger.info(f"\nPredictions only plot successfully saved at : {output_path}")


def save_binary_predictions_plot(
        target_image_path: Path,
        predictions_tensor: tf.Tensor,
        predictions_report_root_path: Path,
):
    # Separate predictions tensor into a list of n_classes binary tensors of size (width, height)
    binary_predictions_sub_dir = predictions_report_root_path / "binary_predictions"
    if not binary_predictions_sub_dir.exists():
        binary_predictions_sub_dir.mkdir()

    # Create and save binary tensors
    for idx, class_number in enumerate(MAPPING_CLASS_NUMBER.values()):
        binary_tensor = tf.where(
            condition=tf.equal(predictions_tensor, class_number),
            x=MASK_TRUE_VALUE,
            y=MASK_FALSE_VALUE,
        )

        binary_tensor_3d = turn_2d_tensor_to_3d_tensor(tensor_2d=binary_tensor)
        mapping_number_class = {
            class_number: class_name
            for class_name, class_number in MAPPING_CLASS_NUMBER.items()
        }

        binary_predictions_class_sub_dir = predictions_report_root_path / "binary_predictions" / get_image_name_without_extension(target_image_path)
        if not binary_predictions_class_sub_dir.exists():
            binary_predictions_class_sub_dir.mkdir(parents=True)
        output_path = (
                binary_predictions_sub_dir
                / get_image_name_without_extension(target_image_path)
                / f"{get_image_name_without_extension(target_image_path)}__{mapping_number_class[idx]}.png"
        )
        tf.keras.preprocessing.image.save_img(output_path, binary_tensor_3d)
        logger.info(f"\nBinary predictions plot successfully saved at : {output_path}")


def save_predictions_config(
        predictions_report_root_path: Path,
        patch_size: int,
        patch_overlap: int,
        batch_size: int,
        encoder_kernel_size: int,
) -> None:
    predictions_config = {
        "patch_size": patch_size,
        "patch_overlap": patch_overlap,
        "batch_size": batch_size,
        "encoder_kernel_size": encoder_kernel_size,
    }

    with open(predictions_report_root_path / "predictions_config.txt", "w") as file:
        # Pass the file handle in as a lambda function to make it callable
        for key, value in predictions_config.items():
            file.write(f"{str(key)}: {str(value)} \n")
