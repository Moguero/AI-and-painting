import pandas as pd
import tensorflow as tf
from loguru import logger
from tensorflow import keras

from constants import (
    N_CLASSES,
    PATCH_SIZE,
    OPTIMIZER,
    LOSS_FUNCTION,
    METRICS,
    REPORTS_ROOT_DIR_PATH,
    N_PATCHES_LIMIT,
    BATCH_SIZE,
    TEST_PROPORTION,
    PATCH_COVERAGE_PERCENT_LIMIT,
    N_EPOCHS,
    PATCHES_DIR_PATH,
    ENCODER_KERNEL_SIZE,
    DATA_AUGMENTATION,
    MAPPING_CLASS_NUMBER,
    PALETTE_HEXA,
)
from dataset_utils.file_utils import timeit, get_formatted_time

# from dataset_utils.files_stats import get_patches_composition
from dataset_utils.files_stats import (
    get_patch_labels_composition,
    get_patches_labels_composition,
)
from dataset_utils.plotting_tools import save_patch_composition_mean_plot
from deep_learning.models.unet import build_small_unet
from dataset_utils.dataset_builder import (
    get_image_patches_paths,
    train_dataset_generator,
    test_dataset_generator,
    get_test_dataset,
)
from pathlib import Path

from deep_learning.training.reporting import make_run_report


@timeit
def train_model(
    n_classes: int,
    patch_size: int,
    optimizer: keras.optimizers,
    loss_function: str,
    metrics: [str],
    report_root_dir_path: Path,
    n_patches_limit: int,
    batch_size: int,
    test_proportion: float,
    patch_coverage_percent_limit: int,
    epochs: int,
    patches_dir_path: Path,
    encoder_kernel_size: int,
    data_augmentation: bool,
    mapping_class_number: {str: int},
    palette_hexa: {int: str},
    add_note: bool = False,
):
    # Add a note in the report to describe the run more specifically
    """
    Build the model, compile it, create a dataset iterator, train the model and save the trained model in callbacks.

    :param n_classes: Number of classes used for the model, background not included.
    :param patch_size: Standard shape of the input images used for the training.
    :param optimizer: Keras optimizer used for compilation.
    :param loss_function: Loss function used for compilation.
    :param metrics: List of metrics used for compilation.
    :param report_root_dir_path: Path of the directory where the reports are stored.
    :param n_patches_limit: Maximum number of patches used for the training.
    :param batch_size: Size of the batches.
    :param test_proportion: Float, used to set the proportion of the test dataset.
    :param patch_coverage_percent_limit: Int, minimum coverage percent of a patch labels on this patch.
    :param epochs: Number of epochs for the training.
    :param patches_dir_path: Path of the main patches directory.
    :param encoder_kernel_size: Tuple of 2 integers, size of the encoder kernel (usually set to 3x3).
    :param data_augmentation: If set to True, perform data augmentation.
    :param mapping_class_number: Mapping dictionary between class names and their representative number.
    :param palette_hexa: Mapping dictionary between class number and their corresponding plotting color.
    :param add_note: If set to True, add a note to the report in order to describe the run shortly.
    :return: The trained model and its metrics history.
    """
    if add_note:
        note = input("Add a note in the report to describe the run more specifically :")
    else:
        note = ""

    # Define the model
    model = build_small_unet(
        n_classes=n_classes,
        input_shape=patch_size,
        batch_size=batch_size,
        encoder_kernel_size=encoder_kernel_size,
    )

    # TEST to define new metrics
    # class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    #     def __init__(self,
    #                  y_true=None,
    #                  y_pred=None,
    #                  num_classes=None,
    #                  name=None,
    #                  dtype=None):
    #         super(UpdatedMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)
    #
    #     def update_state(self, y_true, y_pred, sample_weight=None):
    #         y_pred = tf.math.argmax(y_pred, axis=-1)
    #         return super().update_state(y_true, y_pred, sample_weight)
    #
    # class MyMeanIOU(tf.keras.metrics.MeanIoU):
    #     def update_state(self, y_true, y_pred, sample_weight=None):
    #         return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

    # metrics = [keras.metrics.categorical_accuracy, MyMeanIOU(n_classes)]

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    # Init the callbacks
    # todo : init paths function
    report_dir_path = report_root_dir_path / f"report_{get_formatted_time()}"
    checkpoint_path = (
        report_dir_path / "2_model_report" / "model_checkpoint" / "cp.ckpt"
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, verbose=1, save_weights_only=True
        )
    ]

    # Building dataset
    image_patches_paths_list = get_image_patches_paths(
        patches_dir_path=patches_dir_path,
        n_patches_limit=n_patches_limit,
        batch_size=batch_size,
        patch_coverage_percent_limit=patch_coverage_percent_limit,
        test_proportion=test_proportion,
    )

    # Compute statistics on the dataset
    patches_composition_stats = get_patches_labels_composition(
        image_patches_paths_list=image_patches_paths_list,
        n_classes=n_classes,
        mapping_class_number=mapping_class_number,
    ).describe()

    # Summarize the hyperparameters config used for the training
    model_config = {
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
    }

    # Save a run report
    # todo : put it in the end of the run script to have the history, loss and metrics saved
    make_run_report(
        report_dir_path=report_dir_path,
        model=model,
        model_config=model_config,
        patches_composition_stats=patches_composition_stats,
        palette_hexa=palette_hexa,
        note=note,
    )

    # Fit the model
    logger.info("\nStart model training...")
    # history = model.fit(train_dataset, epochs=epochs, callbacks=callbacks)
    # Warning : the steps_per_epoch param must be not null in order to end the infinite loop of the generator !
    history = model.fit(
        x=train_dataset_generator(
            image_patches_paths=image_patches_paths_list,
            n_classes=n_classes,
            batch_size=batch_size,
            test_proportion=test_proportion,
            data_augmentation=data_augmentation,
        ),
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=int(len(image_patches_paths_list) * (1 - test_proportion))
        // batch_size,
        verbose=1,
    )
    logger.info("\nEnd of model training.")

    # Evaluate the model
    # todo : debug this call of model.evaluate()
    # loss, metrics = model.evaluate(
    #     test_dataset_generator(
    #         image_patches_paths=image_patches_paths_list,
    #         n_classes=n_classes,
    #         batch_size=batch_size,
    #         test_proportion=test_proportion,
    #     ),
    #     callbacks=callbacks,
    # )

    image_tensors_list, labels_tensors_list = get_test_dataset(
        image_patches_paths=image_patches_paths_list,
        n_classes=n_classes,
        batch_size=batch_size,
        test_proportion=test_proportion,
    )

    metrics_values = model.evaluate(
        x=image_tensors_list,
        y=labels_tensors_list,
        callbacks=callbacks,
    )

    # todo : save history and metrics_values

    return model, history, metrics_values
    # return model, history


def load_saved_model(
    checkpoint_dir_path: Path,
    n_classes: int,
    patch_size: int,
    batch_size: int,
    encoder_kernel_size: int,
):
    logger.info("\nLoading the model...")
    model = build_small_unet(n_classes, patch_size, batch_size, encoder_kernel_size)
    filepath = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir_path)
    model.load_weights(filepath=filepath)
    # the warnings logs due to load_weights are here because we don't train (compile/fit) after : they disappear if we do
    logger.info("\nModel loaded successfully.")
    return model


# --------
# DEBUG

# model, history, loss = train_model(N_CLASSES, PATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, METRICS, REPORTS_ROOT_DIR_PATH, N_PATCHES_LIMIT, BATCH_SIZE, TEST_PROPORTION, PATCH_COVERAGE_PERCENT_LIMIT, N_EPOCHS, PATCHES_DIR_PATH, ENCODER_KERNEL_SIZE, DATA_AUGMENTATION, MAPPING_CLASS_NUMBER, PALETTE_HEXA)
# model, history = train_model(N_CLASSES, PATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, METRICS, REPORTS_ROOT_DIR_PATH, N_PATCHES_LIMIT, BATCH_SIZE, TEST_PROPORTION, PATCH_COVERAGE_PERCENT_LIMIT, N_EPOCHS, PATCHES_DIR_PATH, ENCODER_KERNEL_SIZE, DATA_AUGMENTATION, MAPPING_CLASS_NUMBER, PALETTE_HEXA)
# train_model(N_CLASSES, PATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, METRICS, REPORTS_ROOT_DIR_PATHH, N_PATCHES_LIMIT, BATCH_SIZE, TEST_PROPORTION, PATCH_COVERAGE_PERCENT_LIMIT, N_EPOCHS, PATCHES_DIR_PATH, ENCODER_KERNEL_SIZE, DATA_AUGMENTATION, MAPPING_CLASS_NUMBER, PALETTE_HEXA)
# train_model(N_CLASSES, PATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, METRICS, REPORTS_ROOT_DIR_PATH, 100, 16, TEST_PROPORTION, 70, 3, PATCHES_DIR_PATH, ENCODER_KERNEL_SIZE, DATA_AUGMENTATION, MAPPING_CLASS_NUMBER, PALETTE_HEXA)
