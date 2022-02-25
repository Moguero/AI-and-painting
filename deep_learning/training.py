import math
import pandas as pd
from loguru import logger
from tensorflow import keras
from pathlib import Path

from dataset_utils.file_utils import timeit

from dataset_utils.files_stats import (
    get_patches_labels_composition,
)
from deep_learning.unet import build_small_unet
from dataset_utils.dataset_builder import (
    get_image_patches_paths,
    train_dataset_generator,
)
from deep_learning.reporting import (
    build_training_run_report,
    init_report_paths,
)


@timeit
def train_model(
    n_classes: int,
    patch_size: int,
    optimizer: keras.optimizers,
    loss_function: [keras.losses.Loss],
    metrics: [keras.metrics.Metric],
    report_root_dir_path: Path,
    n_patches_limit: int,
    batch_size: int,
    validation_proportion: float,
    test_proportion: float,
    patch_coverage_percent_limit: int,
    epochs: int,
    patches_dir_path: Path,
    encoder_kernel_size: int,
    early_stopping_loss_min_delta: int,
    early_stopping_accuracy_min_delta: int,
    data_augmentation: bool,
    image_data_generator_config_dict: dict,
    mapping_class_number: {str: int},
    palette_hexa: {int: str},
    add_note: bool = False,
    image_patches_paths: [Path] = None,
):
    """
    Build the model, compile it, create a dataset iterator, train the model and save the trained model in callbacks.

    :param early_stopping_accuracy_min_delta: Used in keras.callbacks.EarlyStopping.
    :param early_stopping_loss_min_delta: Used in keras.callbacks.EarlyStopping.
    :param n_classes: Number of classes used for the model, background not included.
    :param patch_size: Standard shape of the input images used for the training.
    :param optimizer: Keras optimizer used for compilation.
    :param loss_function: Loss function used for compilation.
    :param metrics: List of metrics used for compilation.
    :param report_root_dir_path: Path of the directory where the reports are stored.
    :param n_patches_limit: Maximum number of patches used for the training.
    :param batch_size: Size of the batches.
    :param validation_proportion: Float, used to set the proportion of the validation dataset.
    :param test_proportion: Float, used to set the proportion of the test dataset.
    :param patch_coverage_percent_limit: Int, minimum coverage percent of a patch labels on this patch.
    :param epochs: Number of epochs for the training.
    :param patches_dir_path: Path of the main patches directory.
    :param encoder_kernel_size: Tuple of 2 integers, size of the encoder kernel (usually set to 3x3).
    :param data_augmentation: If set to True, perform data augmentation.
    :param image_data_generator_config_dict: :param image_data_generator_config_dict: Dict of parameters to apply with ImageDataGenerator for data augmentation.
    :param mapping_class_number: Mapping dictionary between class names and their representative number.
    :param palette_hexa: Mapping dictionary between class number and their corresponding plotting color.
    :param add_note: If set to True, add a note to the report in order to describe the run shortly.
    :param image_patches_paths: If not None, list of patches to use to make the training on.
    :return: The trained model and its metrics history.
    """

    # Add custom note to the report
    if add_note:
        note = input(
            "\nAdd a note in the report to describe the run more specifically :\n"
        )
    else:
        note = ""

    # Init report paths
    report_paths_dict = init_report_paths(report_root_dir_path=report_root_dir_path)

    # Define the model
    model = build_small_unet(
        n_classes=n_classes,
        input_shape=patch_size,
        batch_size=batch_size,
        encoder_kernel_size=encoder_kernel_size,
    )

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics,
        sample_weight_mode="temporal",
    )

    # Init the callbacks (perform actions at various stages on training)
    callbacks = [
        # save model weights regularly
        keras.callbacks.ModelCheckpoint(
            filepath=report_paths_dict["checkpoint_path"],
            verbose=1,
            save_weights_only=True,
        ),
        # create a dashboard of the training
        keras.callbacks.TensorBoard(
            log_dir=report_paths_dict["model_report"] / "logs",
            update_freq="epoch",
            histogram_freq=1,
        ),
        # stop the training process if the loss stop decreasing considerably
        keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=early_stopping_loss_min_delta,
            patience=1,
            restore_best_weights=True,
        ),
        # stop the training process if the accuracy stop increasing considerably
        keras.callbacks.EarlyStopping(
            monitor=metrics[0].__name__,
            min_delta=early_stopping_accuracy_min_delta,
            patience=1,
            restore_best_weights=True,
        ),
    ]

    # Build training/validation dataset
    image_patches_paths_list = get_image_patches_paths(
        patches_dir_path=patches_dir_path,
        batch_size=batch_size,
        patch_coverage_percent_limit=patch_coverage_percent_limit,
        test_proportion=test_proportion,
        mapping_class_number=mapping_class_number,
        n_patches_limit=n_patches_limit,
        image_patches_paths=image_patches_paths,
    )

    # Compute statistics on the dataset
    patches_composition_stats = get_patches_labels_composition(
        image_patches_paths_list=image_patches_paths_list,
        n_classes=n_classes,
        mapping_class_number=mapping_class_number,
    )

    class_weights_dict = get_class_weights_dict(
        patches_composition_stats=patches_composition_stats,
        mapping_class_number=mapping_class_number,
    )

    # Fit the model
    logger.info("\nStart model training...")
    # Warning : the steps_per_epoch param must be not null in order to end the infinite loop of the generator !
    history = model.fit(
        x=train_dataset_generator(
            image_patches_paths=image_patches_paths_list,
            n_classes=n_classes,
            batch_size=batch_size,
            validation_proportion=validation_proportion,
            test_proportion=test_proportion,
            class_weights_dict=class_weights_dict,
            mapping_class_number=mapping_class_number,
            data_augmentation=data_augmentation,
            image_data_generator_config_dict=image_data_generator_config_dict,
            data_augmentation_plot_path=report_paths_dict[
                "data_augmentation"
            ],  # todo : fix this
        ),
        # class_weight=class_weights_dict,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=int(
            int(len(image_patches_paths_list) * (1 - test_proportion))
            * (1 - validation_proportion)
        )
        // batch_size,
        verbose=1,
    )
    logger.info("\nEnd of model training.")

    # Save a run report
    report_dir_path = report_paths_dict["report_dir_path"]
    build_training_run_report(
        report_dir_path=report_dir_path,
        model=model,
        history=history,
        model_config={
            "n_classes": n_classes,
            "patch_size": patch_size,
            "optimizer": optimizer,
            "loss_function": loss_function,
            "metrics": metrics,
            "n_patches_limit": n_patches_limit,
            "batch_size": batch_size,
            "test_proportion": test_proportion,
            "patch_coverage_percent_limit": patch_coverage_percent_limit,
            "epochs": epochs,
            "encoder_kernel_size": encoder_kernel_size,
            "data_augmentation": data_augmentation,
        },  # summarize the hyperparameters config used for the training
        patches_composition_stats=patches_composition_stats,
        palette_hexa=palette_hexa,
        image_patches_paths_list=image_patches_paths_list,
        class_weights_dict=class_weights_dict,
        mapping_class_number=mapping_class_number,
        note=note,
    )

    return report_dir_path


def get_class_weights_dict(
    patches_composition_stats: pd.DataFrame,
    mapping_class_number: {str: int},
):
    class_proportions_dict = patches_composition_stats.loc["mean"].to_dict()

    class_weights_dict = dict()
    for class_name, class_proportion in class_proportions_dict.items():
        if class_proportion == 0:  # class not in the dataset
            weight = 1
        elif mapping_class_number[class_name] == 0:  # class background
            weight = 1
        else:  # non background class present in the dataset
            # weight = int(1 / class_proportion)
            weight = -int(math.log(class_proportion))

        class_weights_dict[mapping_class_number[class_name]] = weight

    return class_weights_dict


# --------
# DEBUG

# report_dir_path = train_model(N_CLASSES, PATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, METRICS, REPORTS_ROOT_DIR_PATH, N_PATCHES_LIMIT, BATCH_SIZE, VALIDATION_PROPORTION, TEST_PROPORTION, PATCH_COVERAGE_PERCENT_LIMIT, N_EPOCHS, PATCHES_DIR_PATH, ENCODER_KERNEL_SIZE, EARLY_STOPPING_LOSS_MIN_DELTA, EARLY_STOPPING_ACCURACY_MIN_DELTA, DATA_AUGMENTATION, MAPPING_CLASS_NUMBER, PALETTE_HEXA)
