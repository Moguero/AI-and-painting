from loguru import logger
from tqdm import tqdm

from constants import MAPPING_CLASS_NUMBER, MASK_TRUE_VALUE, MASK_FALSE_VALUE
from dataset_utils.file_utils import save_list_to_csv, load_saved_list
from dataset_utils.image_utils import (
    get_image_masks_paths,
    get_mask_class,
    get_image_name_without_extension,
    get_images_paths,
    get_image_patch_masks_paths,
)
from pathlib import Path
import tensorflow as tf
from dataset_utils.image_utils import decode_image


def stack_image_patch_masks(
    image_patch_path: Path,
    save_stats: bool = False,
    save_stats_dir_path: Path = None,
) -> tf.Tensor:
    image_patch_masks_paths = get_image_patch_masks_paths(image_patch_path)
    shape = decode_image(image_patch_masks_paths[0])[:, :, 0].shape
    stacked_tensor = tf.zeros(shape=shape, dtype=tf.int32)

    problematic_indices_list = list()
    zeros_tensor = tf.zeros(shape=shape, dtype=tf.int32)
    for patch_mask_path in image_patch_masks_paths:
        class_categorical_tensor = turn_mask_into_categorical_tensor(patch_mask_path)

        # spotting the problematic pixels indices
        background_mask_class_categorical_tensor = tf.logical_not(
            tf.equal(class_categorical_tensor, zeros_tensor)
        )
        background_mask_stacked_tensor = tf.logical_not(
            tf.equal(stacked_tensor, zeros_tensor)
        )
        logical_tensor = tf.logical_and(
            background_mask_class_categorical_tensor, background_mask_stacked_tensor
        )
        non_overlapping_tensor = tf.equal(
            logical_tensor, tf.constant(False, shape=shape)
        )
        problematic_indices = (
            tf.where(tf.equal(non_overlapping_tensor, False)).numpy().tolist()
        )
        if problematic_indices:
            problematic_indices_list += problematic_indices

        # adding the class mask to the all-classes-stacked tensor
        stacked_tensor = tf.math.add(stacked_tensor, class_categorical_tensor)

    # setting the irregular pixels to background class
    if problematic_indices_list:
        # logger.info(
        #     f"\nMasks are overlapping."
        #     f"\nNumber of problematic pixels : {len(problematic_indices_list)}"
        #     f"\nProblematic pixels indices : {problematic_indices_list}"
        # )
        categorical_array = stacked_tensor.numpy()
        for pixels_coordinates in problematic_indices_list:
            categorical_array[tuple(pixels_coordinates)] = MAPPING_CLASS_NUMBER[
                "background"
            ]
        stacked_tensor = tf.constant(categorical_array, dtype=tf.int32)

    # check that the problematic pixels were set to 0 correctly
    for pixels_coordinates in problematic_indices_list:
        assert (
            stacked_tensor[tuple(pixels_coordinates)].numpy()
            == MAPPING_CLASS_NUMBER["background"]
        )

    # save problematic pixels coordinates
    if save_stats:
        assert (
            save_stats_dir_path is not None
        ), "Please specify a path to save the stats."
        sub_dir = (
            save_stats_dir_path
            / "patches"
            / image_patch_path.parts[-4]
            / image_patch_path.parts[-3]
        )
        if not Path(sub_dir).exists():
            sub_dir.mkdir(parents=True)
            logger.info(f"\nSub folder {sub_dir} was created")
        save_list_to_csv(
            problematic_indices_list,
            sub_dir / f"stats_image_{image_patch_path.parts[-4]}_patch_{image_patch_path.parts[-3]}.csv",
        )

    return stacked_tensor


def one_hot_encode_image_patch_masks(
    image_patch_path: Path, n_classes: int
) -> tf.Tensor:
    categorical_mask_tensor = stack_image_patch_masks(image_patch_path)
    one_hot_encoded_tensor = tf.one_hot(
        categorical_mask_tensor, n_classes + 1, dtype=tf.int32
    )
    return one_hot_encoded_tensor


def turn_mask_into_categorical_tensor(mask_path: Path) -> tf.Tensor:
    tensor_first_channel = decode_image(mask_path)[:, :, 0]
    mask_class = get_mask_class(mask_path)
    categorical_number = MAPPING_CLASS_NUMBER[mask_class]
    categorical_tensor = tf.where(
        tf.equal(tensor_first_channel, MASK_TRUE_VALUE),
        categorical_number,
        MASK_FALSE_VALUE,
    )
    return categorical_tensor


# todo : test write_file and read_file
def save_tensor_to_jpg(tensor: tf.Tensor, output_filepath: Path) -> None:
    file_name = output_filepath.parts[-1]
    assert (
        file_name[-3:] == "jpg"
        or file_name[-3:] == "JPG"
        or file_name[-3:] == "jpeg"
        or file_name[-3:] == "JPEG"
    ), f"The output path {output_filepath} is not with jpg format."
    encoded_image_tensor = tf.io.encode_jpeg(tensor)
    tf.io.write_file(
        filename=tf.constant(str(output_filepath)), contents=encoded_image_tensor
    )


def stack_image_masks(
    image_path: Path,
    masks_dir_path: Path,
) -> tf.Tensor:
    """
    Returns a stacked tensor, with a size corresponding to the number of pixels of image_path.

    :param image_path: The source image on which to compute the stacked labels mask.
    :param masks_dir_path: The masks source directory path.
    :return: A 2D stacked tensor.
    """
    image_masks_paths = get_image_masks_paths(image_path=image_path, masks_dir_path=masks_dir_path)
    shape = decode_image(image_masks_paths[0])[:, :, 0].shape
    stacked_tensor = tf.zeros(shape=shape, dtype=tf.int32)

    problematic_indices_list = list()
    zeros_tensor = tf.zeros(shape=shape, dtype=tf.int32)
    for mask_path in image_masks_paths:
        class_categorical_tensor = turn_mask_into_categorical_tensor(mask_path=mask_path)

        # spotting the problematic pixels indices
        background_mask_class_categorical_tensor = tf.logical_not(
            tf.equal(class_categorical_tensor, zeros_tensor)
        )
        background_mask_stacked_tensor = tf.logical_not(
            tf.equal(stacked_tensor, zeros_tensor)
        )
        logical_tensor = tf.logical_and(
            background_mask_class_categorical_tensor, background_mask_stacked_tensor
        )
        non_overlapping_tensor = tf.equal(
            logical_tensor, tf.constant(False, shape=shape)
        )
        problematic_indices = (
            tf.where(tf.equal(non_overlapping_tensor, False)).numpy().tolist()
        )
        if problematic_indices:
            problematic_indices_list += problematic_indices

        # adding the class mask to the all-classes-stacked tensor
        stacked_tensor = tf.math.add(stacked_tensor, class_categorical_tensor)

    # setting the irregular pixels to background class
    if problematic_indices_list:
        categorical_array = stacked_tensor.numpy()
        for pixels_coordinates in problematic_indices_list:
            categorical_array[tuple(pixels_coordinates)] = MAPPING_CLASS_NUMBER[
                "background"
            ]
        stacked_tensor = tf.constant(categorical_array, dtype=tf.int32)

    # check that the problematic pixels were set to 0 correctly
    for pixels_coordinates in problematic_indices_list:
        assert (
            stacked_tensor[tuple(pixels_coordinates)].numpy()
            == MAPPING_CLASS_NUMBER["background"]
        )

    return stacked_tensor


# todo : delete
def one_hot_encode_image_masks(
    image_path: Path, categorical_masks_dir: Path, n_classes: int
) -> tf.Tensor:
    categorical_mask_tensor = decode_image(get_categorical_mask_path(image_path, categorical_masks_dir))[:, :, 0]
    one_hot_encoded_tensor = tf.one_hot(
        categorical_mask_tensor, n_classes, dtype=tf.int32
    )
    return one_hot_encoded_tensor


# todo : delete
def save_categorical_mask(
    image_path: Path,
    masks_dir: Path,
    output_filepath: Path,
    all_patch_masks_overlap_indices_path: Path,
) -> None:
    """Encode a categorical tensor into a png."""
    stacked_image_tensor = tf.expand_dims(
        tf.cast(
            stack_image_masks(
                image_path, masks_dir, all_patch_masks_overlap_indices_path
            ),
            tf.uint8,
        ),
        -1,
    )
    save_tensor_to_jpg(stacked_image_tensor, output_filepath)
    logger.info(f"\nCategorical mask {output_filepath} was saved successfully.")


# todo: delete
def save_all_categorical_masks(
    images_dir: Path,
    masks_dir: Path,
    categorical_masks_dir: Path,
    all_patch_masks_overlap_indices_path: Path,
) -> None:
    image_dir_paths = get_images_paths(images_dir)
    for image_path in image_dir_paths:
        image_name = get_image_name_without_extension(image_path)
        output_sub_dir = categorical_masks_dir / image_name
        if not output_sub_dir.exists():
            output_sub_dir.mkdir()
            logger.info(f"\nSub folder {output_sub_dir} was created.")
        output_path = output_sub_dir / ("categorical_mask__" + image_name + ".jpg")
        save_categorical_mask(
            image_path, masks_dir, output_path, all_patch_masks_overlap_indices_path
        )


# todo: delete
def get_categorical_mask_path(image_path: Path, categorical_masks_dir: Path):
    image_name = get_image_name_without_extension(image_path)
    categorical_masks_subdir = categorical_masks_dir / image_name
    assert (
        categorical_masks_subdir.exists()
    ), f"Subdir {categorical_masks_subdir} does not exist."
    assert (
        len(list(categorical_masks_subdir.iterdir())) == 1
    ), f"Subdir {categorical_masks_subdir} contains more than one mask."
    return list(categorical_masks_subdir.iterdir())[0]


# debug
def f():
    zeros_tensor = tf.zeros(shape=(3, 3), dtype=tf.int32)
    a = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mask_a = tf.logical_not(tf.equal(a, zeros_tensor))
    b = tf.constant([[0, 0, 0], [2, 2, 0], [0, 0, 0]])
    mask_b = tf.logical_not(tf.equal(b, zeros_tensor))
    logical_tensor = tf.logical_and(mask_a, mask_b)
    non_overlapping_tensor = tf.equal(logical_tensor, tf.constant(False, shape=(3, 3)))
    problematic_indices = (
        tf.where(tf.equal(non_overlapping_tensor, False)).numpy().tolist()
    )
    assert not problematic_indices, (
        f"\nMasks are overlapping."
        f"\nNumber of problematic pixels : {len(problematic_indices)}"
        f"\nProblematic pixels indices : {problematic_indices}"
    )


# debug
def g(SAVE_STATS_DIR_PATH=None):
    problematic_patches = list()
    for image_dir_path in tqdm(SAVE_STATS_DIR_PATH.iterdir(), desc="Iterating through images..."):
        for patch_dir_path in image_dir_path.iterdir():
            for csv_file_path in patch_dir_path.iterdir(): # loop of size 1
                problematic_indices = load_saved_list(csv_file_path)
                if problematic_indices:
                    problematic_patches.append(len(problematic_indices))
    return sum(problematic_patches)