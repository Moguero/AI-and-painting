import numpy as np
from label_utils.mask_loader import load_mask




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


# TODO
def test_image_masks_do_not_overlap(image_path):
    """Checks that the label masks of a given image do not overlap with each other."""

    pass
