import imageio
import numpy as np
import os
import uuid

IMAGE_PATH = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043/_DSC0043.JPG"
MASKS_DIR_PATH = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/"


def load_mask(mask_path: str) -> np.ndarray:
    """Turns png mask into numpy ndarray"""
    mask_array = np.asarray(imageio.imread(mask_path))
    return mask_array


# todo : check if this is still really useful : not sure...
def opacify_mask(mask_array: np.ndarray):
    """Delete the alpha channel of an array by deleting the transparency channel (4th channel).
    As a result, the mask will be fully opaque."""
    return mask_array[:, :, :3]


def get_image_name_without_extension(image_path: str):
    return image_path.split("/")[-1].split(".")[0]


def get_image_name_with_extension(image_path: str):
    return image_path.split("/")[-1]


def get_image_masks_paths(image_path: str, masks_dir: str):
    """Get the paths of the image masks.

    Remark: If the masks sub directory associated to the image does not exist,
    an AssertionError will be thrown."""
    image_masks_sub_dir = masks_dir + get_image_name_without_extension(image_path) + "/"
    assert os.path.exists(image_masks_sub_dir), f"Image masks sub directory {image_masks_sub_dir} does not exist"
    return [
        image_masks_sub_dir + class_mask_sub_dir + "/" + mask_name
        for class_mask_sub_dir in os.listdir(image_masks_sub_dir)
        for mask_name in os.listdir(image_masks_sub_dir + class_mask_sub_dir)
    ]


def merge_masks(image_path: str, masks_dir_path: str):
    """Merges masks of all the classes containing more than one mask."""
    image_masks_sub_dir = masks_dir_path + get_image_name_without_extension(image_path)
    for class_sub_dir in os.listdir(image_masks_sub_dir):
        class_sub_dir_path = image_masks_sub_dir + "/" + class_sub_dir + '/'
        if len(os.listdir(class_sub_dir_path)) > 1:
            masks_to_merge_paths_list = [class_sub_dir_path + class_mask_name for class_mask_name in os.listdir(class_sub_dir_path)]
            output_file_name = class_sub_dir_path + "merged_mask_" + get_image_name_without_extension(image_path) + "_" + class_sub_dir + "__" + uuid.uuid4().hex + ".png"
            create_merged_mask(masks_to_merge_paths_list, output_file_name)


def create_merged_mask(masks_to_merge_paths_list: list, output_file_name: str):
    first_mask = load_mask(masks_to_merge_paths_list[0])
    for class_masks_path_idx in range(1, len(masks_to_merge_paths_list)):
        next_mask = load_mask(masks_to_merge_paths_list[class_masks_path_idx])
        first_mask = first_mask + next_mask
    imageio.imwrite(output_file_name, first_mask)


def get_image_masks(image_path: str, masks_dir: str):
    image_masks_paths = get_image_masks_paths(image_path, masks_dir)
    return [load_mask(mask_path) for mask_path in image_masks_paths]


def get_image_masks_first_channel(image_path: str, masks_dir: str):
    image_masks_paths = get_image_masks_paths(image_path, masks_dir)
    return [load_mask(mask_path)[:, :, 0] for mask_path in image_masks_paths]
