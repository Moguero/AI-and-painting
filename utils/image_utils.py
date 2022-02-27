import random
import shutil
import tensorflow as tf
from typing import Union
from loguru import logger
from pathlib import Path


def decode_image(file_path: Path) -> tf.Tensor:
    """
    Turns a png or jpeg images into its tensor 3D (for jpeg) or 4D (for png) version.

    :param filepath: The path of the file to decode.
    """
    value = tf.io.read_file(str(file_path))
    image_type = get_image_type(file_path)
    channels = get_image_channels_number(file_path)
    if image_type == "png" or image_type == "PNG":
        decoded_image = tf.image.decode_png(value, channels=channels)
    elif (
        image_type == "jpg"
        or image_type == "JPG"
        or image_type == "jpeg"
        or image_type == "JPEG"
    ):
        decoded_image = tf.image.decode_jpeg(value, channels=channels)
    else:
        decoded_image = tf.image.decode_image(value, channels=channels)
    return decoded_image


def get_image_masks_paths(image_path: Path, masks_dir_path: Path) -> [Path]:
    """Get the paths of the images masks.

    Remark: If the masks sub directory associated to the images does not exist,
    an AssertionError will be thrown."""
    image_masks_sub_dir = masks_dir_path / get_image_name_without_extension(image_path)
    assert (
        image_masks_sub_dir.exists()
    ), f"Image masks sub directory {image_masks_sub_dir} does not exist"
    return [
        image_masks_sub_dir / class_mask_sub_dir / mask_name
        for class_mask_sub_dir in image_masks_sub_dir.iterdir()
        for mask_name in (image_masks_sub_dir / class_mask_sub_dir).iterdir()
    ]


def get_image_patch_masks_paths(image_patch_path: Union[Path, str]) -> [Path]:
    """
    Get the paths of the image patch masks.
    """
    if isinstance(image_patch_path, str):
        image_patch_path = Path(image_patch_path)

    image_patch_masks_sub_dir = image_patch_path.parents[1] / "labels"
    assert (
        image_patch_masks_sub_dir.exists()
    ), f"Image patch masks sub directory {image_patch_masks_sub_dir} does not exist"
    return [
        image_patch_masks_sub_dir / class_mask_sub_dir / mask_name
        for class_mask_sub_dir in image_patch_masks_sub_dir.iterdir()
        for mask_name in (image_patch_masks_sub_dir / class_mask_sub_dir).iterdir()
    ]


def get_image_name_without_extension(image_path: Path) -> str:
    return image_path.parts[-1].split(".")[0]


def get_file_name_with_extension(file_path: Path) -> str:
    return file_path.parts[-1]


def get_mask_class(mask_path: Path) -> str:
    return mask_path.parts[-2]


def get_image_type(image_path: Path) -> str:
    return image_path.parts[-1].split(".")[-1]


def get_image_channels_number(image_path: Path) -> int:
    image_type = get_image_type(image_path)
    image_channels_number = None
    if image_type == "png" or image_type == "PNG":
        image_channels_number = 4
    elif (
        image_type == "jpg"
        or image_type == "JPG"
        or image_type == "jpeg"
        or image_type == "JPEG"
    ):
        image_channels_number = 3
    assert (
        image_channels_number is not None
    ), f"Incorrect images type {image_type} of images with path {image_path}"
    return image_channels_number


def get_tensor_dims(tensor: tf.Tensor) -> tuple:
    n_dims = len(list(tensor.shape))
    if n_dims == 2:
        height_index = 0
        width_index = 1
        channels_index = None
    elif n_dims == 3:
        height_index = 0
        width_index = 1
        channels_index = 2
    elif n_dims == 4:
        height_index = 1
        width_index = 2
        channels_index = 3
    else:
        raise ValueError(f"Dimension is {n_dims} : expected 2, 3 or 4.")
    return height_index, width_index, channels_index


def get_image_tensor_shape(image_tensor: tf.Tensor) -> (int, int, int):
    """
    Get the height, width and channels number of an image.
    """
    height_index, width_index, channels_index = get_tensor_dims(tensor=image_tensor)

    image_height = image_tensor.shape[height_index]
    image_width = image_tensor.shape[width_index]
    if channels_index is not None:
        channels_number = image_tensor.shape[channels_index]
    else:
        channels_number = None

    return image_height, image_width, channels_number


def get_dir_paths(dir_path: Path) -> [Path]:
    return list(dir_path.iterdir())


def get_images_paths(images_dir_path: Path) -> [Path]:
    return [
        image_path
        for image_dir_path in images_dir_path.iterdir()
        for image_path in image_dir_path.iterdir()
    ]


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


def get_image_patches_paths_with_limit(
    patches_dir: Path,
    n_patches_limit: int = None,
) -> [Path]:
    """
    Randomly take n_patches_limit patches in the patches_dir folder.
    """
    logger.info("\nRetrieving image patch paths...")
    patch_paths_list = list()
    if n_patches_limit is None:
        for image_patches_subdir in patches_dir.iterdir():
            for patch_dir in image_patches_subdir.iterdir():
                for patch_path in (patch_dir / "image").iterdir():  # loop of size 1
                    patch_paths_list.append(patch_path)
    else:
        counter = 0
        image_patches_subdirs = list(patches_dir.iterdir())
        while counter < n_patches_limit:
            # select a random patch among all the patches
            image_patches_subdir = random.choice(image_patches_subdirs)
            patch_dir = random.choice(list(image_patches_subdir.iterdir()))
            for patch_path in (patch_dir / "image").iterdir():  # loop of size 1
                if patch_path not in patch_paths_list:
                    patch_paths_list.append(patch_path)
                    counter += 1
    logger.info("\nImage patch paths retrieved successfully.")
    return patch_paths_list


def get_files_paths(files_dir: Path) -> [Path]:
    """Get the paths of the files (not folders) in a given folder."""
    file_paths = sorted(
        [
            files_dir / file_name
            for file_name in files_dir.iterdir()
            if file_name.is_file()
        ]
    )
    return file_paths


def copy_image(image_source_path: Path, image_target_dir_path: Path) -> None:
    """Copy an images from a source path to a target path"""
    image_name = get_image_name_without_extension(image_source_path)
    sub_dir = image_target_dir_path / image_name
    file_name = image_name + ".jpg"
    output_path = sub_dir / file_name
    if not Path(sub_dir).exists():
        Path(sub_dir).mkdir()
        logger.info(f"\nSub folder {sub_dir} was created")
    shutil.copyfile(str(image_source_path), str(output_path))


def copy_images_with_masks(
    images_source_dir: Path, masks_dir: Path, image_target_dir_path: Path
):
    """
    Copy/paste images that have a mask in the labels mask directory.

    :param images_source_dir: The folder where the images are stored.
    :param masks_dir: The folder where the images masks are stored.
    :param image_target_dir_path: The folder where to copy the images.
    :return:
    """
    names_of_images_with_masks = get_names_of_images_with_masks(masks_dir)
    for image_path in list(images_source_dir.iterdir()):
        image_name_without_extension = get_image_name_without_extension(image_path)
        if image_name_without_extension in names_of_images_with_masks:
            copy_image(image_path, image_target_dir_path)


def get_names_of_images_with_masks(masks_dir: Path) -> [Path]:
    image_names = [
        mask_image_dir.parts[-1] for mask_image_dir in list(masks_dir.iterdir())
    ]
    return image_names


def group_images_and_all_masks_together(
    dataset_dir: Path, images_dir: Path, masks_dir: Path, categorical_masks_dir: Path
) -> None:
    images_dirs_paths = get_dir_paths(images_dir)
    for image_dir_path in images_dirs_paths:
        image_dir_name = image_dir_path.parts[-1]
        dataset_subdir = dataset_dir / image_dir_name
        if not dataset_subdir.exists():
            dataset_subdir.mkdir()
            logger.info(f"\nSub folder {dataset_subdir} was created.")

        # Copy all the images to dataset_dir
        for image_path in image_dir_path.iterdir():  # loop of size 1
            image_name_with_extension = get_file_name_with_extension(image_path)
            output_path = dataset_subdir / image_name_with_extension
            shutil.copyfile(str(image_path), str(output_path))

        # Copy all the categorical masks to dataset_dir
        categorical_mask_dir_path = categorical_masks_dir / image_dir_name
        for (
            categorical_mask_path
        ) in categorical_mask_dir_path.iterdir():  # loop of size 1
            output_path = dataset_subdir / get_file_name_with_extension(
                categorical_mask_path
            )
            shutil.copyfile(str(categorical_mask_path), str(output_path))

        # Copy all the binary masks to dataset_dir
        masks_dir_path = masks_dir / image_dir_name
        for class_dir_path in masks_dir_path.iterdir():  # loop of size 9 maximum
            for mask_path in class_dir_path.iterdir():
                output_path = dataset_subdir / get_file_name_with_extension(mask_path)
                shutil.copyfile(str(mask_path), str(output_path))
