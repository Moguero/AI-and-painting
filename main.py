import argparse
import warnings
from pathlib import Path

from constants import (
    N_CLASSES,
    PATCH_SIZE,
    OPTIMIZER,
    LOSS_FUNCTION,
    METRICS,
    REPORTS_ROOT_DIR_PATH,
    N_PATCHES_LIMIT,
    BATCH_SIZE,
    VALIDATION_PROPORTION,
    TEST_PROPORTION,
    PATCH_COVERAGE_PERCENT_LIMIT,
    N_EPOCHS,
    PATCHES_DIR_PATH,
    ENCODER_KERNEL_SIZE,
    DATA_AUGMENTATION,
    MAPPING_CLASS_NUMBER,
    PALETTE_HEXA,
    PATCH_OVERLAP,
    # TEST_IMAGES_PATHS_LIST,
    DOWNSCALED_TEST_IMAGES_PATHS_LIST,
    EARLY_STOPPING_LOSS_MIN_DELTA,
    EARLY_STOPPING_ACCURACY_MIN_DELTA,
    CORRELATE_PREDICTIONS_BOOL, CORRELATION_FILTER, IMAGE_DATA_GENERATOR_CONFIG_DICT,
)
from deep_learning.training.model_runner import train_model
from deep_learning.training.reporting import build_predict_run_report


def main(
    train_bool: bool,
    predict_bool: bool,
    light_report_bool: bool,
    add_note: bool,
    n_patches_limit: int,
    n_epochs: int,
    report_dir: str,
    data_augmentation: bool
) -> None:
    if train_bool:
        report_dir_path = train_model(
            n_classes=N_CLASSES,
            patch_size=PATCH_SIZE,
            optimizer=OPTIMIZER,
            loss_function=LOSS_FUNCTION,
            metrics=METRICS,
            report_root_dir_path=REPORTS_ROOT_DIR_PATH,
            n_patches_limit=n_patches_limit,
            batch_size=BATCH_SIZE,
            validation_proportion=VALIDATION_PROPORTION,
            test_proportion=TEST_PROPORTION,
            patch_coverage_percent_limit=PATCH_COVERAGE_PERCENT_LIMIT,
            epochs=n_epochs,
            patches_dir_path=PATCHES_DIR_PATH,
            encoder_kernel_size=ENCODER_KERNEL_SIZE,
            early_stopping_loss_min_delta=EARLY_STOPPING_LOSS_MIN_DELTA,
            early_stopping_accuracy_min_delta=EARLY_STOPPING_ACCURACY_MIN_DELTA,
            data_augmentation=data_augmentation,
            image_data_generator_config_dict=IMAGE_DATA_GENERATOR_CONFIG_DICT,
            mapping_class_number=MAPPING_CLASS_NUMBER,
            palette_hexa=PALETTE_HEXA,
            add_note=add_note,
        )

        if predict_bool:
            build_predict_run_report(
                test_images_paths_list=DOWNSCALED_TEST_IMAGES_PATHS_LIST,
                report_dir_path=report_dir_path,
                patch_size=PATCH_SIZE,
                patch_overlap=PATCH_OVERLAP,
                n_classes=N_CLASSES,
                batch_size=BATCH_SIZE,
                encoder_kernel_size=ENCODER_KERNEL_SIZE,
                light_report_bool=light_report_bool,
                correlate_predictions_bool=CORRELATE_PREDICTIONS_BOOL,
                correlation_filter=CORRELATION_FILTER,
            )
    else:  # case no training
        if predict_bool:
            if report_dir is None:
                report_dir = input(
                    "Please specify a correct report directory path. \nEx: .../reports/report_2021_12_13__13_12_18\n"
                )
            report_dir_path = Path(report_dir)
            if not report_dir_path.exists():
                raise ValueError(
                    f"This report directory path does no exist : {report_dir_path}"
                )

            build_predict_run_report(
                test_images_paths_list=DOWNSCALED_TEST_IMAGES_PATHS_LIST,
                report_dir_path=report_dir_path,
                patch_size=PATCH_SIZE,
                patch_overlap=PATCH_OVERLAP,
                n_classes=N_CLASSES,
                batch_size=BATCH_SIZE,
                encoder_kernel_size=ENCODER_KERNEL_SIZE,
                light_report_bool=light_report_bool,
                correlate_predictions_bool=CORRELATE_PREDICTIONS_BOOL,
                correlation_filter=CORRELATION_FILTER,
            )

        else:
            raise ValueError(
                "Nothing happened since parameter --train and --predict were set to False"
            )


if __name__ == "__main__":
    # Parser setup
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument(
        "--train",
        "-t",
        help="Whether to create and train a new model.",
        action="store_true",
    )
    parser.add_argument(
        "--predict",
        "-pred",
        help="Whether to infer predictions on test images. Will ask the user a model path to use.",
        action="store_true",
    )
    parser.add_argument(
        "--light",
        "-l",
        help="Build a light report with image/predictions comparisons only. Should only be used with --predict.",
        action="store_true",
    )
    parser.add_argument(
        "--note",
        "-n",
        help="Add a note to the training report. Should only be used with --train.",
        action="store_true",
    )
    parser.add_argument(
        "--patches-limit",
        "-pl",
        type=int,
        help="Maximum number of patches to use for training. Should only be used with --train.",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        default=N_EPOCHS,
        type=int,
        help="Maximum number of epochs during training. Should only be used with --train.",
    )

    parser.add_argument(
        "--report",
        "-r",
        help="Report path to give the model to make the inference with. Should only be used with --predict.",
    )
    parser.add_argument(
        "--data-augment",
        "-da",
        action="store_true",
        help="Apply data augmentation on the training data. Should only be used with --train.",
    )
    args = parser.parse_args()

    if not args.predict and not args.train:
        raise ValueError(
            "At least one of --train or --predict parameters should be given."
        )

    if not args.train and args.note:
        warnings.warn("--note parameter should only be used with --train parameter")

    if not args.train and args.patches_limit != N_PATCHES_LIMIT:
        warnings.warn(
            "--patches-limit parameter should only be used with --train parameter"
        )

    if not args.train and args.epochs != N_EPOCHS:
        warnings.warn("--epochs parameter should only be used with --train parameter")

    if not args.train and args.epochs is True:
        warnings.warn("--data-augment parameter should only be used with --train parameter")

    if not args.predict and args.light:
        warnings.warn("--light parameter should only be used with --predict parameter.")

    if not args.predict and args.report is not None:
        warnings.warn(
            "--report parameter should only be used with --predict parameter."
        )

    main(
        train_bool=args.train,
        predict_bool=args.predict,
        light_report_bool=args.light,
        add_note=args.note,
        n_patches_limit=args.patches_limit,
        n_epochs=args.epochs,
        report_dir=args.report,
        data_augmentation=args.data_augment,
    )


# CLI command
# python main.py --train --predict

# todo : faire une liste d'improvements à la fin du README

# todo : store number of patches used for training
