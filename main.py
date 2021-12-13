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
    TEST_IMAGE_PATH,
    PATCH_OVERLAP, DOWNSCALE_FACTORS,
)
from deep_learning.inference.predictions_maker import make_predictions, save_predictions_plot_only
from deep_learning.training.model_runner import train_model


def main():
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

    save_predictions_plot_only(
        target_image_path=TEST_IMAGE_PATH,
        report_dir_path=report_dir_path,
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP,
        n_classes=N_CLASSES,
        batch_size=BATCH_SIZE,
        encoder_kernel_size=ENCODER_KERNEL_SIZE,
        downscale_factors=DOWNSCALE_FACTORS,
    )


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
