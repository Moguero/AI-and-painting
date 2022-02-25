import ast
import tensorflow as tf
from pathlib import Path

from constants import MAPPING_CLASS_NUMBER
from dataset_utils.image_utils import get_image_patch_masks_paths, decode_image
from dataset_utils.masks_encoder import turn_mask_into_categorical_tensor


def stack_image_patch_masks_reformatted(
    image_patch_path: Path, all_patch_masks_overlap_indices_dict: dict
) -> tf.Tensor:
    """Fetches the images mask first channels, transform it into a categorical tensor and add the categorical together."""
    image_masks_paths = get_image_patch_masks_paths(image_patch_path)
    shape = decode_image(image_masks_paths[0])[:, :, 0].shape
    stacked_tensor = tf.zeros(shape=shape, dtype=tf.int32)
    for mask_path in image_masks_paths:
        categorical_tensor = turn_mask_into_categorical_tensor(mask_path)
        stacked_tensor = tf.math.add(stacked_tensor, categorical_tensor)

    # setting the irregular pixels to background class
    problematic_indices_list = ast.literal_eval(
        all_patch_masks_overlap_indices_dict[str(image_patch_path)]
    )["problematic_indices"]
    if problematic_indices_list:
        categorical_array = stacked_tensor.numpy()
        for pixels_coordinates in problematic_indices_list:
            categorical_array[
                pixels_coordinates[0], pixels_coordinates[1]
            ] = MAPPING_CLASS_NUMBER["background"]
        stacked_tensor = tf.constant(categorical_array, dtype=tf.int32)

    return stacked_tensor
