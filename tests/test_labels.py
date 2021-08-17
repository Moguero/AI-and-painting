import json
from pathlib import Path
import numpy as np
import tensorflow as tf

from constants import MAPPING_CLASS_NUMBER, MASK_FALSE_VALUE, MASK_TRUE_VALUE
from dataset_utils.files_stats import count_mask_value_occurences_of_2d_tensor
from dataset_utils.image_utils import (
    decode_image,
    get_dir_paths,
)
from dataset_utils.image_utils import get_image_masks_paths
from labelbox_utils.mask_downloader import get_full_json

IMAGE_PATH = Path(r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files\images\_DSC0030\_DSC0030.jpg")
MASKS_DIR = Path(r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files\labels_masks\all")
IMAGES_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/")
CATEGORICAL_MASKS_DIR = Path(
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/categorical_masks/"
)
MASK_PATH = Path(
    r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\labels_masks\all\1\bois-tronc\mask_1_bois-tronc__73261f79c8e54f7b9102e01b3ebc65bf.png"
)
JSON_PATH = Path(
    "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/labelbox_export_json/export-2021-07-26T14_40_28.059Z.json"
)
CATEGORICAL_MASK_PATH = Path(
    r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\categorical_masks\1\categorical_mask__1.jpg"
)


# todo : make those tests run automatically in a shell script


def test_mask_channels_are_equal(mask_path: Path) -> None:
    """Checks that all the mask channels are equal, to be sure that the mask is channel independent."""
    mask_array = decode_image(mask_path).numpy()
    for channel_number in (1, 2, 3):
        assert np.array_equal(
            mask_array[:, :, channel_number], mask_array[:, :, 0]
        ), "The channels of this mask are not equal."


def test_mask_first_channel_is_binary(mask_path: Path) -> None:
    """Checks that the mask first channel is binary, i.e. contains only 0 or 255 values.

    Remark : we only tests on the mask first channel,
    supposing we will run test_mask_channels_are_equal() function later."""
    mask_array = decode_image(mask_path).numpy()
    assert set(np.unique(mask_array[:, :, 0])) == {
        0,
        255,
    }, "Mask is not binary, i.e. contains other values than 0 or 255."


def test_image_masks_do_not_overlap(image_path: Path, masks_dir: Path) -> None:
    """Checks that the label masks of a given images do not overlap with each other.
    It supposes that the mask has already been checked as binary (with 0 and 255 values only)."""
    image_masks_paths = get_image_masks_paths(image_path, masks_dir)
    if len(image_masks_paths) != 1:
        shape = decode_image(image_masks_paths[0])[:, :, 0].shape
        stacked_tensor = tf.zeros(shape=shape, dtype=tf.int32)
        for mask_path in image_masks_paths:
            stacked_tensor = tf.math.add(
                stacked_tensor, tf.cast(decode_image(mask_path)[:, :, 0], tf.int32)
            )
        stacked_values = list(
            tf.unique_with_counts(tf.reshape(stacked_tensor, [-1])).y.numpy()
        )
        assert set(stacked_values) - {MASK_FALSE_VALUE, MASK_TRUE_VALUE} == {}, (
            f"\nSome masks overlap with each other."
            f"\nValues with counts are : {count_mask_value_occurences_of_2d_tensor(stacked_tensor)}"
            f"\nProblematic indices : {[tf.where(tf.equal(stacked_tensor, value)).numpy() for value in set(stacked_values) - {MASK_FALSE_VALUE, MASK_TRUE_VALUE}]}"
        )
        # assert 255 not in image_first_mask, "Some masks overlap with each other."


# todo : test that the mask is inside the picture


def test_minimum_one_mask_per_image(json_path: str) -> None:
    """Checks if there is at least one mask url per images in the json path."""
    number_of_masks_per_image = dict()
    with open(json_path) as f:
        json_dict = json.load(f)
        for image in json_dict:
            external_id = image["External ID"]
            if bool(image["Label"]):
                image_objects = image["Label"]["objects"]
                number_of_masks_per_image[external_id] = len(image_objects)
            else:
                number_of_masks_per_image[external_id] = 0
    assert (
        0 not in number_of_masks_per_image.values()
    ), f"The following images have no mask : {[external_id for external_id, masks_number in number_of_masks_per_image.items() if masks_number == 0]}"


def test_one_mask_per_class_only(masks_dir: Path) -> None:
    for image_dir in masks_dir.iterdir():
        for image_class_dir in image_dir.iterdir():
            assert (
                len(list(image_class_dir.iterdir())) == 1
            ), f"Image {image_dir} has no or more than one class mask for class {image_class_dir}."


def test_value_equals_title(json_path: Path) -> None:
    for image_dict in get_full_json(json_path):
        for object in image_dict["Label"]["objects"]:
            assert (
                object["title"] == object["value"]
            ), f"Image {image_dict['External ID']} has title {object['title']} different than value {object['value']}"


def test_same_folders_of_images_and_masks(images_dir: Path, masks_dir: Path) -> None:
    assert (
        get_dir_paths(images_dir).sort() == get_dir_paths(masks_dir).sort()
    ), f"These folders are in {images_dir} but not in {masks_dir} :  {[item for item in get_dir_paths(images_dir) not in get_dir_paths(masks_dir).sort()]}"


def test_same_folders_of_images_and_categorical_masks(
    images_dir: Path, categorical_masks_dir: Path
) -> None:
    assert (
        get_dir_paths(images_dir).sort() == get_dir_paths(categorical_masks_dir).sort()
    ), f"These folders are in {images_dir} but not in {categorical_masks_dir} :  {[item for item in get_dir_paths(images_dir) not in get_dir_paths(categorical_masks_dir).sort()]}"


def test_categorical_mask_values(categorical_mask_path: Path):
    mask_first_channel = decode_image(categorical_mask_path)[:, :, 0]
    unique_with_count_tensor = tf.unique_with_counts(
        tf.reshape(mask_first_channel, [-1])
    )
    values_list = list(unique_with_count_tensor.y.numpy())
    assert sorted(values_list) == sorted(
        MAPPING_CLASS_NUMBER.values()
    ), f"\nValues {[value for value in values_list if value not in MAPPING_CLASS_NUMBER.values()]} are not existing values."
