import os

IMAGE_PATH = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/angular_logo.png"
LABELS_MASKS_DIR = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/"


def get_image_name_without_extension(image_path: str):
    return image_path.split("/")[-1].split(".")[0]


def get_image_name_with_extension(image_path: str):
    return image_path.split("/")[-1]


# todo : personaliser les messages d'erreurs des assert
def get_image_masks_paths(image_path: str, labels_mask_dir: str):
    """Get the paths of the image masks.

    Remark: If the masks sub directory associated to the image does not exist,
    an AssertionError will be thrown."""
    image_masks_sub_dir = labels_mask_dir + get_image_name_without_extension(image_path)
    assert os.path.exists(image_masks_sub_dir), "Image masks sub directory does not exist"
    return [
        labels_mask_dir + get_image_name_without_extension(image_path) + "/" + mask_name
        for mask_name in os.listdir(image_masks_sub_dir)
    ]
