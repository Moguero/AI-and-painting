import math
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from tensorflow import keras
from pathlib import Path
from typing import Generator, Tuple

from dataset_builder.masks_encoder import (
    one_hot_encode_image_patch_masks,
    stack_image_patch_masks,
)
from utils.image_utils import (
    decode_image,
    get_image_patch_masks_paths,
)
from utils.plotting_utils import save_image_from_tensor
from utils.time_utils import timeit
from utils.files_stats import (
    get_patches_labels_composition,
    get_image_patches_paths,
)
from deep_learning.unet import build_small_unet
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


def train_dataset_generator(
    image_patches_paths: [Path],
    n_classes: int,
    batch_size: int,
    validation_proportion: float,
    test_proportion: float,
    class_weights_dict: {int: float},
    mapping_class_number: {str: int},
    data_augmentation: bool = False,
    image_data_generator_config_dict: dict = {},
    data_augmentation_plot_path: Path = None,
) -> Generator[Tuple[tf.Tensor, tf.Tensor], None, None]:
    """
    Create a dataset generator that will be used to feed model.fit().
    This generator yields batches (acts like "drop_remainder == True") of augmented images.

    Warning : this function builds a generator which is only meant to be used with "model.fit()" function.
    Appropriate behaviour is not expected in a different context.

    :param image_patches_paths: Paths of the images (already filtered) to train on.
    :param n_classes: Number of classes, background not included.
    :param batch_size: Size of the batches.
    :param validation_proportion: Float, used to set the proportion of the validation images dataset.
    :param test_proportion: Float, used to set the proportion of the training images dataset.
    :param class_weights_dict: Mapping of classes and their global weight in the dataset : used to balance the loss function.
    :param mapping_class_number: Mapping dictionary between class names and their representative number.
    :param data_augmentation: Boolean, apply data augmentation to the each batch of the training dataset if True.
    :param image_data_generator_config_dict: Dict of parameters to apply with ImageDataGenerator for data augmentation.
    :param data_augmentation_plot_path: Path where to store the intermediate augmented patches samples.
    :return: Yield 2 tensors of size (batch_size, patch_size, patch_size, 3) and (batch_size, patch_size, patch_size, n_classes + 1),
            corresponding to image tensors and their corresponding one-hot-encoded masks tensors
    """
    assert sorted(class_weights_dict) == [
        i for i in range(n_classes + 1)
    ], f"Class weights dict is missing a class : should have {n_classes + 1} keys but is {class_weights_dict}"

    validation_limit_idx = int(len(image_patches_paths) * (1 - test_proportion))
    train_limit_idx = int(validation_limit_idx * (1 - validation_proportion))
    n_batches = train_limit_idx // batch_size

    logger.info(
        f"\n{validation_limit_idx}/{len(image_patches_paths)} patches taken for training and validation : validation proportion of {validation_proportion} and test proportion of {test_proportion}"
        f"\n - {train_limit_idx}/{len(image_patches_paths)} patches for training (validation proportion of {validation_proportion} and test proportion of {test_proportion})"
        f"\n - {validation_limit_idx - train_limit_idx}/{len(image_patches_paths)} patches for validation"
        f"\n{(train_limit_idx // batch_size) * batch_size}/{train_limit_idx} training patches will be kept and {train_limit_idx % batch_size}/{train_limit_idx} will be dropped (drop remainder)."
        f"\n{n_batches} batches taken for training"
    )

    save_augmented_patches_count = True
    while True:
        for n_batch in range(n_batches):

            # lists of length batch_size, containing :
            # - image tensors of shape (patch_size, patch_size, 3)
            # - labels tensors of shape (patch_size, patch_size, n_classes + 1)
            # - weights tensors of shape (patch_size, patch_size, n_classes + 1)
            image_tensors_list = list()
            labels_tensors_list = list()
            weights_tensors_list = list()

            for image_patch_path in image_patches_paths[
                n_batch * batch_size : (n_batch + 1) * batch_size
            ]:
                image_tensor = decode_image(file_path=image_patch_path)
                image_patch_masks_paths = get_image_patch_masks_paths(
                    image_patch_path=image_patch_path
                )
                labels_tensor = one_hot_encode_image_patch_masks(
                    image_patch_masks_paths=image_patch_masks_paths,
                    n_classes=n_classes,
                    mapping_class_number=mapping_class_number,
                )
                weights_tensor = get_image_weights(
                    image_patch_masks_paths=image_patch_masks_paths,
                    mapping_class_number=mapping_class_number,
                    class_weights_dict=class_weights_dict,
                )

                image_tensors_list.append(image_tensor)
                labels_tensors_list.append(labels_tensor)
                weights_tensors_list.append(weights_tensor)

            if data_augmentation:
                if save_augmented_patches_count is False:
                    data_augmentation_plot_path = None

                image_tensors, labels_tensors, weights_tensors = augment_batch(
                    image_tensors=image_tensors_list,
                    labels_tensors=labels_tensors_list,
                    weights_tensors=weights_tensors_list,
                    batch_size=batch_size,
                    image_data_generator_config_dict=image_data_generator_config_dict,
                    data_augmentation_plot_path=data_augmentation_plot_path,
                )
                save_augmented_patches_count = False  # only save the first batch plot
            else:
                image_tensors = tf.stack(values=image_tensors_list)
                labels_tensors = tf.stack(values=labels_tensors_list)
                weights_tensors = tf.stack(values=weights_tensors_list)

            yield image_tensors, labels_tensors, weights_tensors


def get_image_weights(
    image_patch_masks_paths: [Path],
    mapping_class_number: {str: int},
    class_weights_dict: {int: int},
):
    categorical_mask_array = stack_image_patch_masks(
        image_patch_masks_paths=image_patch_masks_paths,
        mapping_class_number=mapping_class_number,
    ).numpy()
    vectorize_function = np.vectorize(lambda x: class_weights_dict[x])
    weights_array = vectorize_function(categorical_mask_array)
    weights_tensor = tf.constant(value=weights_array)
    return weights_tensor


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


def augment_batch(
    image_tensors: [tf.Tensor],
    labels_tensors: [tf.Tensor],
    weights_tensors: [tf.Tensor],
    batch_size: int,
    image_data_generator_config_dict: dict,
    data_augmentation_plot_path: Path = None,
):
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        **image_data_generator_config_dict
    )
    seed = 1
    image_data_generator.fit(image_tensors, augment=True, seed=seed)

    for (
        augmented_image_array,
        augmented_labels_array,
        augmented_weights_array,
    ) in image_data_generator.flow(
        x=tf.stack(image_tensors),
        y=tf.stack(labels_tensors),
        batch_size=batch_size,
        sample_weight=tf.stack(weights_tensors),
        shuffle=False,
    ):
        augmented_image_tensors = tf.constant(augmented_image_array, dtype=tf.int32)
        augmented_labels_tensors = tf.constant(augmented_labels_array, dtype=tf.int32)
        augmented_weights_tensors = tf.constant(augmented_weights_array, dtype=tf.int32)
        break

    if data_augmentation_plot_path is not None:
        for idx in range(batch_size):
            if idx == 0:
                concat_images_tensor = image_tensors[0]
                concat_augmented_images_tensor = tf.cast(
                    augmented_image_tensors[0], dtype=tf.uint8
                )
            else:
                concat_images_tensor = tf.concat(
                    [concat_images_tensor, image_tensors[idx]], axis=1
                )
                concat_augmented_images_tensor = tf.concat(
                    [
                        concat_augmented_images_tensor,
                        tf.cast(augmented_image_tensors[idx], dtype=tf.uint8),
                    ],
                    axis=1,
                )
        comparison_tensor = tf.concat(
            [concat_images_tensor, concat_augmented_images_tensor], axis=0
        )
        # plot_image_from_tensor(tensor=comparison_tensor)
        save_image_from_tensor(
            tensor=comparison_tensor,
            output_path=data_augmentation_plot_path,
            title="An augmented batch sample.",
        )

    return augmented_image_tensors, augmented_labels_tensors, augmented_weights_tensors


# --------
# DEBUG

# report_dir_path = train_model(N_CLASSES, PATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, METRICS, REPORTS_ROOT_DIR_PATH, N_PATCHES_LIMIT, BATCH_SIZE, VALIDATION_PROPORTION, TEST_PROPORTION, PATCH_COVERAGE_PERCENT_LIMIT, N_EPOCHS, PATCHES_DIR_PATH, ENCODER_KERNEL_SIZE, EARLY_STOPPING_LOSS_MIN_DELTA, EARLY_STOPPING_ACCURACY_MIN_DELTA, DATA_AUGMENTATION, MAPPING_CLASS_NUMBER, PALETTE_HEXA)
