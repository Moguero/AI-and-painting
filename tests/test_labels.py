import json
from pathlib import Path
import numpy as np

from dataset_utils.image_utils import decode_image
from dataset_utils.image_utils import get_image_masks_paths

IMAGE_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043/_DSC0043.JPG")
MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/")
MASK_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/_DSC0043/feuilles_vertes/mask__DSC0043_feuilles_vertes__3466c2cda646448fbe8f4927f918e247.png")
JSON_PATH = Path("C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/labelbox_export_json/export-2021-07-12T14_48_36.100Z.json")


def test_mask_channels_are_equal(mask_path: Path) -> None:
    """Checks that all the mask channels are equal, to be sure that the mask is channel independent."""
    mask_array = decode_image(mask_path).numpy()
    for channel_number in (1, 2, 3):
        assert np.array_equal(mask_array[:, :, channel_number], mask_array[:, :, 0]), "The channels of this mask are not equal."


def test_mask_first_channel_is_binary(mask_path: Path) -> None:
    """Checks that the mask first channel is binary, i.e. contains only 0 or 255 values.

    Remark : we only tests on the mask first channel,
    supposing we will run test_mask_channels_are_equal() function later."""
    mask_array = decode_image(mask_path).numpy()
    assert set(np.unique(mask_array[:, :, 0])) == {0, 255}, "Mask is not binary, i.e. contains other values than 0 or 255."


def test_image_masks_do_not_overlap(image_path: Path, masks_dir: Path) -> None:
    """Checks that the label masks of a given image do not overlap with each other.
    It supposes that the mask has already been checked as binary (with 0 and 255 values only)."""
    image_masks_paths = get_image_masks_paths(image_path, masks_dir)
    if len(image_masks_paths) != 1:
        image_first_mask = decode_image(image_masks_paths[0]).numpy()
        for mask_number in range(1, len(image_masks_paths)):
            image_first_mask = image_first_mask & decode_image(image_masks_paths[mask_number]).numpy()
        assert 255 not in image_first_mask, "Some masks overlap with each other."


# todo : test that the mask is inside the picture

def test_minimum_one_mask_per_image(json_path: str) -> None:
    """Checks if there is at least one mask url per image in the json path."""
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


# todo : check that for each object in the json, the value is equal to the title

# todo : implement a check that there is just one mask per class : list the images with several : they will have to be merged
