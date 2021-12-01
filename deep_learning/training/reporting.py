from pathlib import Path

import pandas as pd
from tensorflow import keras

from dataset_utils.plotting_tools import save_patch_composition_mean_plot


def make_run_report(
    report_dir_path: Path,
    model: keras.Model,
    model_config: dict,
    patches_composition_stats: pd.DataFrame,
    palette_hexa: {int: str},
    note: str,
):
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

    # Export


    # 3. predictions

    # image_name of the predictions
    return
