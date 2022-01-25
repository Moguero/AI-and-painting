from pathlib import Path
from typing import List

import matplotlib
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf

from constants import (
    PALETTE_HEXA,
    MAPPING_CLASS_NUMBER,
    MASK_TRUE_VALUE,
    MASK_FALSE_VALUE,
    TEST_IMAGES_PATHS_LIST,
)
from dataset_utils.file_utils import get_formatted_time
from dataset_utils.image_utils import (
    decode_image,
    get_image_name_without_extension,
    get_image_tensor_shape,
)
from dataset_utils.plotting_tools import (
    save_patch_composition_plot,
    map_categorical_mask_to_3_color_channels_tensor,
    turn_2d_tensor_to_3d_tensor,
)
from deep_learning.inference.predictions_maker import (
    make_predictions,
)
from deep_learning.postprocessing.median_filtering import (
    save_median_filtering_comparison,
)


def build_training_run_report(
    report_dir_path: Path,
    model: keras.Model,
    history: keras.callbacks.History,
    model_config: dict,
    patches_composition_stats: pd.DataFrame,
    palette_hexa: {int: str},
    image_patches_paths_list: [Path],
    class_weights_dict: {int: int},
    mapping_class_number: {str: int},
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

    # Plot original patches labels composition
    patches_composition_mean_stats_dict = patches_composition_stats.loc[
        "mean"
    ].to_dict()
    patch_composition_mean_plot_output_path = (
        report_subdirs_paths_dict["data_report"] / "patches_composition.png"
    )
    save_patch_composition_plot(
        patch_composition_stats_dict=patches_composition_mean_stats_dict,
        output_path=patch_composition_mean_plot_output_path,
        palette_hexa=palette_hexa,
    )

    # Save the figures with which the original composition plot was made
    patch_original_composition_output_path = (
        report_subdirs_paths_dict["data_report"] / "patches_composition.txt"
    )
    with open(patch_original_composition_output_path, "w") as file:
        for key, value in patches_composition_mean_stats_dict.items():
            file.write(f"{key}: {value},\n")

    # Plot rebalanced patches labels composition
    patches_composition_mean_stats_dict = patches_composition_stats.loc[
        "mean"
    ].to_dict()
    patches_rebalanced_composition_stats_dict = dict()
    classes_weights_sum = 0
    for class_name in patches_composition_mean_stats_dict.keys():
        classes_weights_sum += (
            patches_rebalanced_composition_stats_dict[class_name]
            * class_weights_dict[mapping_class_number[class_name]]
        )

    for class_name in patches_composition_mean_stats_dict.keys():
        patches_rebalanced_composition_stats_dict[class_name] = (
            patches_composition_mean_stats_dict[class_name]
            * class_weights_dict[mapping_class_number[class_name]]
        ) / classes_weights_sum
    patch_composition_mean_plot_output_path = (
        report_subdirs_paths_dict["data_report"] / "rebalanced_patches_composition.png"
    )
    save_patch_composition_plot(
        patch_composition_stats_dict=patches_rebalanced_composition_stats_dict,
        output_path=patch_composition_mean_plot_output_path,
        palette_hexa=palette_hexa,
    )

    # Save the figures with which the rebalanced composition plot was made
    patch_rebalanced_composition_output_path = (
        report_subdirs_paths_dict["data_report"] / "rebalanced_patches_composition.txt"
    )
    with open(patch_rebalanced_composition_output_path, "w") as file:
        for key, value in patches_rebalanced_composition_stats_dict.items():
            file.write(f"{key}: {value},\n")

    # Save patches used for training
    patches_paths_path = report_subdirs_paths_dict["data_report"] / "patches_paths.txt"
    with open(patches_paths_path, "w") as file:
        file.write("[\n")
        for patch_path in image_patches_paths_list:
            file.write(f"{patch_path},\n")
        file.write("]")

    # todo : plot minimized patches used (array of patches)

    # 2. model report

    # Save the model architecture
    model_architecture_filename = (
        report_subdirs_paths_dict["model_report"] / "model_architecture.txt"
    )
    with open(model_architecture_filename, "w") as file:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: file.write(x + "\n"))

    # Save the model History.history object (with loss and metrics history)
    history_filename = report_subdirs_paths_dict["model_report"] / "history.txt"
    with open(history_filename, "w") as file:
        file.write(f"{history.history}")

    # Save the README that describes how to open Tensor Board
    readme_filename = report_subdirs_paths_dict["model_report"] / "README.md"
    with open(readme_filename, "w") as file:
        file.write(
            f"To start TensorBoard, run the following command line in a terminal :"
            f"\n\ntensorboard --logdir='{str(report_subdirs_paths_dict['model_report'] / 'logs')}'"
            f"\n\nThen, open a web browser and go to :"
            f"\n\n http://localhost:6006/ "
        )

    # Save the model config (hyperparameters)
    model_config_filename = (
        report_subdirs_paths_dict["model_report"] / "model_config.txt"
    )
    with open(model_config_filename, "w") as file:
        for key, value in model_config.items():
            file.write(f"{str(key)}: {str(value)} \n")


def init_report_paths(report_root_dir_path: Path) -> {str: Path}:
    _report_dir_path = report_root_dir_path / f"report_{get_formatted_time()}"
    _data_report = _report_dir_path / "1_data_report"
    _model_report = _report_dir_path / "2_model_report"
    _predictions_report = _report_dir_path / "3_predictions_report"
    _checkpoint_path = _model_report / "model_checkpoint/model_checkpoint"

    report_paths_dict = dict()
    report_paths_dict["report_dir_path"] = _report_dir_path
    report_paths_dict["data_report"] = _data_report
    report_paths_dict["model_report"] = _model_report
    report_paths_dict["predictions_report"] = _predictions_report
    report_paths_dict["checkpoint_path"] = _checkpoint_path

    return report_paths_dict


def build_predict_run_report(
    test_images_paths_list: List[Path],
    report_dir_path: Path,
    patch_size: int,
    patch_overlap: int,
    n_classes: int,
    batch_size: int,
    encoder_kernel_size: int,
    light_report_bool: bool,
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

        save_predictions_config(
            predictions_report_root_path=predictions_report_root_path,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            batch_size=batch_size,
            encoder_kernel_size=encoder_kernel_size,
        )

        if not light_report_bool:

            predictions_only_path = save_predictions_only_plot(
                target_image_path=test_image_path,
                predictions_tensor=predictions_tensor,
                predictions_report_root_path=predictions_report_root_path,
            )

            save_binary_predictions_plot(
                target_image_path=test_image_path,
                predictions_tensor=predictions_tensor,
                predictions_report_root_path=predictions_report_root_path,
            )

            save_median_filtering_comparison(
                source_image_path=predictions_only_path,
                predictions_report_root_path=predictions_report_root_path,
            )


def save_test_images_vs_predictions_plot(
    target_image_path: Path,
    predictions_tensor: tf.Tensor,
    predictions_report_root_path: Path,
) -> None:
    # Set-up plotting settings
    image_tensor = decode_image(file_path=target_image_path)
    target_image_height, target_image_width, channels_number = get_image_tensor_shape(
        image_tensor=image_tensor
    )
    image = image_tensor.numpy()

    (
        predictions_tensor_height,
        predictions_tensor_width,
        channels_number,
    ) = get_image_tensor_shape(image_tensor=predictions_tensor)
    mapped_predictions_array = map_categorical_mask_to_3_color_channels_tensor(
        categorical_mask_tensor=predictions_tensor
    )

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(
        f"Image : {get_image_name_without_extension(image_path=target_image_path)}"
    )
    ax1.set_title(f"Original image ({target_image_width}x{target_image_height})")
    ax2.set_title(
        f"Predictions ({predictions_tensor_width}x{predictions_tensor_height})"
    )
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
) -> Path:
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

    return output_path


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

        binary_predictions_class_sub_dir = (
            predictions_report_root_path
            / "binary_predictions"
            / get_image_name_without_extension(target_image_path)
        )
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
