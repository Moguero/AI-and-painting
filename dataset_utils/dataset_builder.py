import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Generator, Tuple, Union, List

from loguru import logger
from tqdm import tqdm

from dataset_utils.image_utils import (
    decode_image,
    get_image_patches_paths_with_limit,
    get_image_patch_masks_paths,
)
from dataset_utils.masks_encoder import (
    one_hot_encode_image_patch_masks,
    stack_image_patch_masks,
)
from dataset_utils.patches_generator import extract_patches
from dataset_utils.patches_sorter import get_patch_coverage
from dataset_utils.plotting_tools import (
    plot_image_from_tensor,
    save_image_from_tensor,
)


def get_image_patches_paths(
    patches_dir_path: Path,
    batch_size: int,
    patch_coverage_percent_limit: int,
    test_proportion: float,
    mapping_class_number: {str: int},
    n_patches_limit: int = None,
    image_patches_paths: [Path] = None,
) -> [Path]:
    """
    Get images patches paths on which the model will train on.
    Also filter the valid patches above the coverage percent limit.

    :param patches_dir_path: Path of the patches root folder.
    :param n_patches_limit: Maximum number of patches used for the training.
    :param batch_size: Size of the batches.
    :param patch_coverage_percent_limit: Int, minimum coverage percent of a patch labels on this patch.
    :param test_proportion: Float, used to set the proportion of the test dataset.
    :param mapping_class_number: Mapping dictionary between class names and their representative number.
    :param image_patches_paths: If not None, list of patches to use to make the training on.
    :return: A list of paths of images to train on.
    """
    assert (
        0 <= test_proportion < 1
    ), f"Test proportion must be between 0 and 1 : {test_proportion} was given."
    logger.info("\nStart to build dataset...")
    assert (n_patches_limit // batch_size) * (
        1 - test_proportion
    ) >= 1, f"Size of training dataset is 0. Increase the n_patches_limit parameter or decrease the batch_size."

    logger.info("\nGet the paths of the valid patches for training...")
    patches_under_coverage_percent_limit_list = list()

    # if the image patches to train on are not provided, randomly select n_patches_limit patches
    if image_patches_paths is None:
        image_patches_paths = get_image_patches_paths_with_limit(
            patches_dir=patches_dir_path, n_patches_limit=n_patches_limit
        )

    for image_patch_path in tqdm(
        image_patches_paths,
        desc="Selecting patches above the coverage percent limit...",
    ):
        image_patch_masks_paths = get_image_patch_masks_paths(
            image_patch_path=image_patch_path
        )
        coverage_percent = get_patch_coverage(
            image_patch_masks_paths=image_patch_masks_paths,
            mapping_class_number=mapping_class_number,
        )
        if int(float(coverage_percent)) > patch_coverage_percent_limit:
            patches_under_coverage_percent_limit_list.append(image_patch_path)

    logger.info(
        f"\n{len(patches_under_coverage_percent_limit_list)}/{len(image_patches_paths)} patches above coverage percent limit selected."
    )
    return patches_under_coverage_percent_limit_list


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

    # todo : data augmentation with ImageDataGenerator : will create a generator,
    #  that can be used to reload the augmented images in a tensor that will be yielded by this generator
    #  pass an augmentation_object parameter to this function (cf tuto)
    #  random crop, luminosité, légère rotation, hue, saturation, compression jpeg
    #  hsv cylinder
    #  algo de traitement de signal jpeg pour la compression d'images
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    # idée : data augmentation on each batch
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


# NOT USED ANYMORE
# def validation_dataset_generator(
#     image_patches_paths: [Path],
#     n_classes: int,
#     batch_size: int,
#     validation_proportion: float,
#     test_proportion: float,
#     mapping_class_number: {str: int},
#     data_augmentation: bool = False,
# ) -> Generator[Tuple[tf.Tensor, tf.Tensor], None, None]:
#     """
#     Create a dataset generator that will be used to feed model.fit().
#     This generator yields batches (acts like "drop_remainder == True") of augmented images.
#
#     Warning : this function builds a generator which is only meant to be used with "model.fit()" function.
#     Appropriate behaviour is not expected in a different context.
#
#     :param image_patches_paths: Paths of the images (already filtered) to train on.
#     :param n_classes: Number of classes, background not included.
#     :param batch_size: Size of the batches.
#     :param validation_proportion: Float, used to set the proportion of the validation images dataset.
#     :param test_proportion: Float, used to set the proportion of the training images dataset.
#     :param mapping_class_number: Mapping dictionary between class names and their representative number.
#     :param data_augmentation: Boolean, apply data augmentation to the each batch of the training dataset if True.
#     :return: Yield 2 tensors of size (batch_size, patch_size, patch_size, 3) and (batch_size, patch_size, patch_size, n_classes + 1),
#             corresponding to image tensors and their corresponding one-hot-encoded masks tensors
#     """
#     validation_limit_idx = int(len(image_patches_paths) * (1 - test_proportion))
#     train_limit_idx = int(validation_limit_idx * (1 - validation_proportion))
#
#     while True:
#         n_batches = (validation_limit_idx - train_limit_idx) // batch_size
#         for n_batch in range(n_batches):
#             # lists of length batch_size, containing :
#             # - image tensors of shape (patch_size, patch_size, 3)
#             # - labels tensors of shape (patch_size, patch_size, n_classes + 1)
#             # - weights tensors of shape (patch_size, patch_size, n_classes + 1)
#             image_tensors_list = list()
#             labels_tensors_list = list()
#             weights_tensors_list = list()
#
#             for image_patch_path in image_patches_paths[
#                 train_limit_idx
#                 + n_batch * batch_size : train_limit_idx
#                 + (n_batch + 1) * batch_size
#             ]:
#                 image_tensor, labels_tensor = decode_image(
#                     file_path=image_patch_path
#                 ), one_hot_encode_image_patch_masks(
#                     image_patch_path=image_patch_path,
#                     n_classes=n_classes,
#                     mapping_class_number=mapping_class_number,
#                 )
#                 weights_tensor = ...  # todo
#
#                 image_tensors_list.append(image_tensor)
#                 labels_tensors_list.append(labels_tensor)
#                 weights_tensors_list.append(weights_tensor)
#
#             if data_augmentation:
#                 (
#                     augmented_image_tensors,
#                     augmented_labels_tensors,
#                     weights_tensors,
#                 ) = augment_batch(
#                     image_tensors=image_tensors_list,
#                     labels_tensors=labels_tensors_list,
#                     weights_tensors=weights_tensors_list,
#                     batch_size=batch_size,
#                 )
#             else:
#                 augmented_image_tensors, augmented_labels_tensors = tf.stack(
#                     image_tensors_list
#                 ), tf.stack(labels_tensors_list)
#
#             yield augmented_image_tensors, augmented_labels_tensors


# def test_dataset_generator(
#     image_patches_paths: [Path],
#     n_classes: int,
#     batch_size: int,
#     test_proportion: float,
# ) -> Generator[Tuple[tf.Tensor, tf.Tensor], None, None]:
#     """
#     Create a dataset generator that will be used to feed model.evaluate()
#     . This generator yields batches (acts like "drop_remainder == True") of augmented images.
#
#     Warning : this function builds a generator which is only meant to be used with "model.evaluate()" function.
#     Appropriate behaviour is not expected in a different context.
#
#     :param image_patches_paths: Paths of the images (already filtered) to train on.
#     :param n_classes: Number of classes, background not included.
#     :param batch_size: Size of the batches.
#     :param test_proportion: Float, used to set the proportion of the validation images dataset.
#     :return: Yield 2 tensors of size (batch_size, patch_size, patch_size, 3) and (batch_size, patch_size, patch_size, n_classes + 1),
#             corresponding to image tensors and their corresponding one-hot-encoded masks tensors
#     """
#     train_limit_idx = int(len(image_patches_paths) * (1 - test_proportion))
#
#     while True:
#         n_batches = (len(image_patches_paths) - train_limit_idx) // batch_size
#         for n_batch in range(n_batches):
#             # list of length batch_size, containing image tensors of shape (256, 256, 3)
#             image_tensors_list = list()
#             # list of length batch_size, containing corresponding labels tensors of shape (256, 256, 10)
#             labels_tensors_list = list()
#
#             for image_patch_path in image_patches_paths[
#                 train_limit_idx
#                 + n_batch * batch_size : train_limit_idx
#                 + (n_batch + 1) * batch_size
#             ]:
#                 image_tensor, labels_tensor = decode_image(
#                     file_path=image_patch_path
#                 ), one_hot_encode_image_patch_masks(
#                     image_patch_path=image_patch_path, n_classes=n_classes
#                 )
#                 image_tensors_list.append(image_tensor)
#                 labels_tensors_list.append(labels_tensor)
#             yield tf.stack(image_tensors_list), tf.stack(labels_tensors_list)


def build_predictions_dataset(
    target_image_tensor: tf.Tensor, patch_size: int, patch_overlap: int
) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
    """
    Build a dataset of patches to make predictions on each one of them.

    :param target_image_tensor: Image to make predictions on.
    :param patch_size: Size of the patch.
    :param patch_overlap: Number of pixels on which neighbors patches intersect each other.
    :return: A Dataset object with tensors of size (1, patch_size, patch_size, 3). Its length corresponds of the number of patches generated.
    """
    logger.info("\nSlice the image into patches...")
    (
        main_patches_tensors_list,
        right_side_patches_tensors_list,
        down_side_patches_tensors_list,
    ) = extract_patches(
        image_tensor=target_image_tensor,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )
    prediction_dataset = tf.data.Dataset.from_tensor_slices(main_patches_tensors_list)
    prediction_dataset = prediction_dataset.batch(
        batch_size=1, drop_remainder=True
    )  # todo : don't hardcode the batch_size to 1

    right_side_prediction_dataset = tf.data.Dataset.from_tensor_slices(
        right_side_patches_tensors_list
    )
    right_side_prediction_dataset = right_side_prediction_dataset.batch(
        batch_size=1, drop_remainder=True
    )

    down_side_prediction_dataset = tf.data.Dataset.from_tensor_slices(
        down_side_patches_tensors_list
    )
    down_side_prediction_dataset = down_side_prediction_dataset.batch(
        batch_size=1, drop_remainder=True
    )

    logger.info(f"\n{len(prediction_dataset)} patches created successfully.")
    logger.info(
        f"\n{len(right_side_prediction_dataset)} right side patches created successfully."
    )
    logger.info(
        f"\n{len(down_side_prediction_dataset)} down side patches created successfully."
    )

    return (
        prediction_dataset,
        right_side_prediction_dataset,
        down_side_prediction_dataset,
    )


# deprecated + never finished
def get_train_dataset(
    n_patches_limit: int,
    n_classes: int,
    batch_size: int,
    validation_proportion: float,
    test_proportion: float,
    class_weights_dict: {int: float},
    patch_coverage_percent_limit: int,
    patches_dir_path: Path,
    mapping_class_number: {str: int},
) -> tf.data.Dataset:
    """

    :param n_patches_limit: Maximum number of patches used for the training.
    :param n_classes: Total number of classes, background not included.
    :param batch_size: Size of the batch.
    :param validation_proportion: Float, used to set the proportion of the validation images dataset.
    :param test_proportion: Float, used to set the proportion of the test dataset.
    :param class_weights_dict: Mapping of classes and their global weight in the dataset : used to balance the loss function.
    :param patch_coverage_percent_limit: Int, minimum coverage percent of a patch labels on this patch.
    :param patches_dir_path: Path of the folder with patches of size patch_size.
    :param mapping_class_number: Mapping dictionary between class names and their representative number.
    :return: A training Dataset object of length (n_patches_limit // batch_size) * (1 - test_proportion),
        with tuples of tensors of size (batch_size, patch_size, patch_size, 3) and
        (batch_size, patch_size, patch_size, n_classes + 1) respectfully.
    """
    assert (
        0 <= test_proportion < 1
    ), f"Test proportion must be between 0 and 1 : {test_proportion} was given."
    logger.info("\nStart to build dataset...")
    assert (n_patches_limit // batch_size) * (
        1 - test_proportion
    ) >= 1, f"Size of training dataset is 0. Increase the n_patches_limit parameter or decrease the batch_size."
    assert sorted(class_weights_dict) == [
        i for i in range(n_classes + 1)
    ], f"Class weights dict is missing a class : should have {n_classes + 1} keys but is {class_weights_dict}"

    # Get the paths of the valid patches for training
    image_patches_paths = get_image_patches_paths(
        patches_dir_path=patches_dir_path,
        patch_coverage_percent_limit=patch_coverage_percent_limit,
        batch_size=batch_size,
        test_proportion=test_proportion,
        mapping_class_number=mapping_class_number,
        n_patches_limit=n_patches_limit,
    )

    image_patches_masks_path = [
        get_image_patch_masks_paths(image_patch_path)
        for image_patch_path in image_patches_paths
    ]

    validation_limit_idx = int(len(image_patches_paths) * (1 - test_proportion))
    train_limit_idx = int(validation_limit_idx * (1 - validation_proportion))

    logger.info(
        f"\n{validation_limit_idx}/{len(image_patches_paths)} patches taken for training and validation : validation proportion of {validation_proportion} and test proportion of {test_proportion}"
        f"\n - {train_limit_idx}/{len(image_patches_paths)} patches for training (validation proportion of {validation_proportion} and test proportion of {test_proportion})"
        f"\n - {validation_limit_idx - train_limit_idx}/{len(image_patches_paths)} patches for validation"
        f"\n{(train_limit_idx // batch_size) * batch_size}/{train_limit_idx} training patches will be kept and {train_limit_idx % batch_size}/{train_limit_idx} will be dropped (drop remainder).",
    )

    # image_patches_paths = [str(i) for i in image_patches_paths]
    # image_patches_masks_path = [
    #     get_image_patch_masks_paths(image_patch_path)
    #     for image_patch_path in image_patches_paths
    # ]
    # patches_image_and_labels_paths: List[Tuple[str, List[str]]] = [
    #     (str(image_patch_path), get_image_patch_masks_paths(image_patch_path))
    #     for image_patch_path in image_patches_paths
    # ]
    #
    # # dataset = tf.data.Dataset.list_files(image_patches_paths)
    # dataset = tf.data.Dataset.from_generator(
    #     lambda: patches_image_and_labels_paths, tf.string
    # )
    #
    # def process_path(patches_image_and_labels_paths) -> (tf.Tensor, tf.Tensor):
    #     breakpoint()
    #     # Load the raw data from the file as a string
    #     image_tensor = tf.io.read_file(patches_image_and_labels_paths[0])
    #     image_tensor = tf.io.decode_image(image_tensor, channels=3)
    #
    #     labels_tensor = tf.io.read_file(patches_image_and_labels_paths[0])
    #     labels_tensor = one_hot_encode_image_patch_masks(
    #         image_patch_masks_paths=patches_image_and_labels_paths[1],
    #         n_classes=n_classes,
    #         mapping_class_number=mapping_class_number,
    #     )
    #     # Resize the image to the desired size
    #     return image_tensor
    #
    # dataset = dataset.map(process_path)
    # breakpoint()

    # Create image patches tensors and their associated one-hot-encoded masks tensors
    image_tensors = [
        decode_image(Path(image_path))
        for image_path in tqdm(image_patches_paths, desc="Loading image tensors")
    ]
    mask_tensors = [
        one_hot_encode_image_patch_masks(
            image_patch_path=Path(image_patch_path),
            n_classes=n_classes,
            mapping_class_number=mapping_class_number,
        )
        for image_patch_path in tqdm(image_patches_paths, desc="Loading mask tensors")
    ]

    # Create a tf.data.Dataset object and batch it
    full_dataset = tf.data.Dataset.from_tensor_slices((image_tensors, mask_tensors))
    full_dataset = full_dataset.batch(batch_size, drop_remainder=True)
    # todo : shuffle the whole dataset

    # Split the dataset into a training and testing one
    train_limit_idx = int(len(full_dataset) * (1 - test_proportion))
    train_dataset = full_dataset.take(train_limit_idx)

    logger.info("\nDataset built successfully.")
    return train_dataset


def augment_batch(
    image_tensors: [tf.Tensor],
    labels_tensors: [tf.Tensor],
    weights_tensors: [tf.Tensor],
    batch_size: int,
    image_data_generator_config_dict: dict,
    data_augmentation_plot_path: str = None,
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
