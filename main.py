import argparse
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
    TEST_IMAGES_PATHS_LIST,
    DOWNSCALED_TEST_IMAGES_PATHS_LIST,
)
from deep_learning.training.model_runner import train_model
from deep_learning.training.reporting import build_predict_run_report


def main(
    train_bool: bool,
    predict_bool: bool,
) -> None:
    if train_bool:
        model, history, report_dir_path = train_model(
            n_classes=N_CLASSES,
            patch_size=PATCH_SIZE,
            optimizer=OPTIMIZER,
            loss_function=LOSS_FUNCTION,
            metrics=METRICS,
            report_root_dir_path=REPORTS_ROOT_DIR_PATH,
            n_patches_limit=N_PATCHES_LIMIT,
            batch_size=BATCH_SIZE,
            validation_proportion=VALIDATION_PROPORTION,
            test_proportion=TEST_PROPORTION,
            patch_coverage_percent_limit=PATCH_COVERAGE_PERCENT_LIMIT,
            epochs=N_EPOCHS,
            patches_dir_path=PATCHES_DIR_PATH,
            encoder_kernel_size=ENCODER_KERNEL_SIZE,
            data_augmentation=DATA_AUGMENTATION,
            mapping_class_number=MAPPING_CLASS_NUMBER,
            palette_hexa=PALETTE_HEXA,
        )

        if predict_bool:
            # todo : specify the image path in the parser OR create a config.json
            build_predict_run_report(
                test_images_paths_list=DOWNSCALED_TEST_IMAGES_PATHS_LIST,
                report_dir_path=report_dir_path,
                patch_size=PATCH_SIZE,
                patch_overlap=PATCH_OVERLAP,
                n_classes=N_CLASSES,
                batch_size=BATCH_SIZE,
                encoder_kernel_size=ENCODER_KERNEL_SIZE,
            )
    else:  # case no training
        if predict_bool:
            report_dir_path = Path(
                input(
                    "What is the report directory path ? \nEx: .../reports/report_2021_12_13__13_12_18\n"
                )
            )
            if not report_dir_path.exists():
                raise ValueError("This report directory path does no exist.")

            # todo : specify the report path in the parser OR create a config.json
            build_predict_run_report(
                test_images_paths_list=DOWNSCALED_TEST_IMAGES_PATHS_LIST,
                report_dir_path=report_dir_path,
                patch_size=PATCH_SIZE,
                patch_overlap=PATCH_OVERLAP,
                n_classes=N_CLASSES,
                batch_size=BATCH_SIZE,
                encoder_kernel_size=ENCODER_KERNEL_SIZE,
            )

        else:
            raise ValueError(
                "Nothing happened since parameter --train and --predict were set to False"
            )


if __name__ == "__main__":
    # Parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", default="false", help="Which library to use : can be pandas or dask"
    )
    parser.add_argument(
        "--predict",
        default="false",
        help="Which library to use : can be pandas or dask",
    )
    args = parser.parse_args()

    if args.train.lower() == "true":
        train_bool = True
    elif args.train.lower() == "false":
        train_bool = False
    else:
        raise ValueError(
            "train optional argument must be equal to True, true, False or false"
        )

    if args.predict.lower() == "true":
        predict_bool = True
    elif args.predict.lower() == "false":
        predict_bool = False
    else:
        raise ValueError(
            f"predict optional argument must be equal to True, true, False or false : {args.predict} was given"
        )

    main(
        train_bool=train_bool,
        predict_bool=predict_bool,
    )


# CLI command
# python main.py --train true --predict true
