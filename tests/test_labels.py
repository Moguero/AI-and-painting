import numpy as np
from label_utils.mask_loader import load_mask
from image_utils.image_and_masks import get_image_masks_paths

IMAGE_PATH = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/angular_logo.png"
IMAGE_PATH = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/airflow_logo.png"
IMAGE_PATH = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/AVZD6310.png"
LABELS_MASKS_DIR = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/"


def test_mask_channels_are_equal(mask_path: str):
    """Checks that all the mask channels are equal, to be sure that the mask is channel independent."""
    mask_array = load_mask(mask_path)
    for channel_number in (1, 2, 3):
        assert np.array_equal(mask_array[:, :, channel_number], mask_array[:, :, 0])


def test_mask_first_channel_is_binary(mask_path: str):
    """Checks that the mask first channel is binary, i.e. contains only 0 or 255 values.

    Remark : we only tests on the mask first channel,
    supposing we will run test_mask_channels_are_equal() function later."""
    mask_array = load_mask(mask_path)
    assert set(np.unique(mask_array[:, :, 0])) == {0, 255}


def test_image_masks_do_not_overlap(image_path: str, labels_mask_dir: str):
    """Checks that the label masks of a given image do not overlap with each other.
    It supposes that the mask has already been checked as binary."""
    image_masks_paths = get_image_masks_paths(image_path, labels_mask_dir)
    masks_arrays = [load_mask(mask_path) for mask_path in image_masks_paths]
    if len(masks_arrays) != 1:
        boolean_array = masks_arrays[0]
        for mask_number in range(1, len(masks_arrays)):
            boolean_array = boolean_array & masks_arrays[mask_number]
        assert True in boolean_array, "Some masks overlap with each other."
