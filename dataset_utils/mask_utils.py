import imageio
import numpy as np
from pathlib import Path

IMAGE_PATH = Path(
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043/_DSC0043.JPG"
)
MASKS_DIR_PATH = Path(
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks"
)


# todo : replace load_mask by decode_image in the code base
def load_mask(mask_path: Path) -> np.ndarray:
    """Turns png mask into numpy ndarray"""
    mask_array = np.asarray(imageio.imread(mask_path))
    return mask_array


# todo : check if this is still really useful : not sure...
def opacify_mask(mask_array: np.ndarray) -> np.ndarray:
    """Delete the alpha channel of an array by deleting the transparency channel (4th channel).
    As a result, the mask will be fully opaque."""
    return mask_array[:, :, :3]


def get_image_name_without_extension(image_path: Path) -> str:
    return image_path.parts[-1].split(".")[0]


def get_image_name_with_extension(image_path: Path) -> str:
    return image_path.parts[-1]


def get_image_masks_paths(image_path: Path, masks_dir: Path) -> [Path]:
    """Get the paths of the image masks.

    Remark: If the masks sub directory associated to the image does not exist,
    an AssertionError will be thrown."""
    image_masks_sub_dir = masks_dir / get_image_name_without_extension(image_path)
    assert (
        image_masks_sub_dir.exists()
    ), f"Image masks sub directory {image_masks_sub_dir} does not exist"
    return [
        image_masks_sub_dir / class_mask_sub_dir / mask_name
        for class_mask_sub_dir in image_masks_sub_dir.iterdir()
        for mask_name in (image_masks_sub_dir / class_mask_sub_dir).iterdir()
    ]
