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
    VALIDATION_PROPORTION,
)
from dataset_utils.file_utils import timeit

from dataset_utils.files_stats import (
    get_patches_labels_composition,
)
from deep_learning.models.unet import build_small_unet, build_small_unet_arbitrary_input
from dataset_utils.dataset_builder import (
    get_image_patches_paths,
    train_dataset_generator,
)
from pathlib import Path

from deep_learning.training.reporting import (
    build_training_run_report,
    init_report_paths,
)


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
    validation_proportion: float,
    test_proportion: float,
    patch_coverage_percent_limit: int,
    epochs: int,
    patches_dir_path: Path,
    encoder_kernel_size: int,
    early_stopping_loss_min_delta: int,
    data_augmentation: bool,
    mapping_class_number: {str: int},
    palette_hexa: {int: str},
    add_note: bool = False,
    image_patches_paths: [Path] = None,
):
    # Add a note in the report to describe the run more specifically
    """
    Build the model, compile it, create a dataset iterator, train the model and save the trained model in callbacks.

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
    :param mapping_class_number: Mapping dictionary between class names and their representative number.
    :param palette_hexa: Mapping dictionary between class number and their corresponding plotting color.
    :param add_note: If set to True, add a note to the report in order to describe the run shortly.
    :param image_patches_paths: If not None, list of patches to use to make the training on.
    :return: The trained model and its metrics history.
    """

    # Add custom note to the report
    if add_note:
        note = input("Add a note in the report to describe the run more specifically :")
    else:
        note = ""

    # Init report paths
    report_paths_dict = init_report_paths(report_root_dir_path=report_root_dir_path)
    # todo : check repo to see the function to init path + use of json configs

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
    ]

    # Build training/validation dataset
    image_patches_paths_list = get_image_patches_paths(
        patches_dir_path=patches_dir_path,
        n_patches_limit=n_patches_limit,
        batch_size=batch_size,
        patch_coverage_percent_limit=patch_coverage_percent_limit,
        test_proportion=test_proportion,
        image_patches_paths=image_patches_paths,
    )

    # Compute statistics on the dataset
    patches_composition_stats = get_patches_labels_composition(
        image_patches_paths_list=image_patches_paths_list,
        n_classes=n_classes,
        mapping_class_number=mapping_class_number,
    ).describe()

    # Fit the model
    logger.info("\nStart model training...")
    # history = model.fit(train_dataset, epochs=epochs, callbacks=callbacks)
    # Warning : the steps_per_epoch param must be not null in order to end the infinite loop of the generator !
    history = model.fit(
        x=train_dataset_generator(
            image_patches_paths=image_patches_paths_list,
            n_classes=n_classes,
            batch_size=batch_size,
            validation_proportion=validation_proportion,
            test_proportion=test_proportion,
            data_augmentation=data_augmentation,
        ),
        # validation_data=validation_dataset_generator(
        #     image_patches_paths=image_patches_paths_list,
        #     n_classes=n_classes,
        #     batch_size=batch_size,
        #     validation_proportion=validation_proportion,
        #     test_proportion=test_proportion,
        #     data_augmentation=data_augmentation,
        # ),
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

    # image_tensors_list, labels_tensors_list = get_test_dataset(
    #     image_patches_paths=image_patches_paths_list,
    #     n_classes=n_classes,
    #     batch_size=batch_size,
    #     test_proportion=test_proportion,
    # )
    #
    # metrics_values = model.evaluate(
    #     x=image_tensors_list,
    #     y=labels_tensors_list,
    #     callbacks=callbacks,
    # )

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
        note=note,
    )

    # return model, report_dir_path, metrics_values
    return report_dir_path


# --------
# DEBUG

# model, history = train_model(N_CLASSES, PATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, METRICS, REPORTS_ROOT_DIR_PATH, N_PATCHES_LIMIT, BATCH_SIZE, VALIDATION_PROPORTION, TEST_PROPORTION, PATCH_COVERAGE_PERCENT_LIMIT, N_EPOCHS, PATCHES_DIR_PATH, ENCODER_KERNEL_SIZE, DATA_AUGMENTATION, MAPPING_CLASS_NUMBER, PALETTE_HEXA)
# model, history, metrics = train_model(N_CLASSES, PATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, METRICS, REPORTS_ROOT_DIR_PATH, N_PATCHES_LIMIT, BATCH_SIZE, VALIDATION_PROPORTION, TEST_PROPORTION, PATCH_COVERAGE_PERCENT_LIMIT, N_EPOCHS, PATCHES_DIR_PATH, ENCODER_KERNEL_SIZE, DATA_AUGMENTATION, MAPPING_CLASS_NUMBER, PALETTE_HEXA)
# train_model(N_CLASSES, PATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, METRICS, REPORTS_ROOT_DIR_PATH, N_PATCHES_LIMIT, BATCH_SIZE, VALIDATION_PROPORTION, TEST_PROPORTION, PATCH_COVERAGE_PERCENT_LIMIT, N_EPOCHS, PATCHES_DIR_PATH, ENCODER_KERNEL_SIZE, DATA_AUGMENTATION, MAPPING_CLASS_NUMBER, PALETTE_HEXA)
# train_model(N_CLASSES, PATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, METRICS, REPORTS_ROOT_DIR_PATH, 100, 16, VALIDATION_PROPORTION, TEST_PROPORTION, 70, 3, PATCHES_DIR_PATH, ENCODER_KERNEL_SIZE, DATA_AUGMENTATION, MAPPING_CLASS_NUMBER, PALETTE_HEXA)
