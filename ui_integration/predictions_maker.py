from pathlib import Path
from ui_integration.model import load_saved_model
from ui_integration.utils import decode_image, get_image_tensor_shape, get_image_name_without_extension, \
    get_file_name_with_extension

import numpy as np
import tensorflow as tf


def make_predictions(
    target_image_path: Path,
    checkpoint_dir_path: Path,
    patch_size: int,
    patch_overlap: int,
    n_classes: int,
    batch_size: int,
    encoder_kernel_size: int,
    misclassification_size: int = 5,
) -> tf.Tensor:
    """
    Make predictions on the target image specified with its path.

    :param encoder_kernel_size: Size of the kernel encoder.
    :param target_image_path: Image to make predictions on.
    :param checkpoint_dir_path: Path of the already trained model.
    :param patch_size: Size of the patches on which the model was trained. This is also the size of the predictions patches.
    :param patch_overlap: Number of pixels on which neighbors patches intersect each other.
    :param n_classes: Number of classes to map, background excluded.
    :param batch_size: Batch size that was used for the model which is loaded.
    :param correlate_predictions_bool: Whether or not to apply correlation on the predictions, i.e. make a local weighted mean on each class probability.
    :param correlation_filter: The filter to use for correlation.
    :param misclassification_size: Estimated number of pixels on which the classification is wrong due to side effects between neighbors patches.

    :return: A 2D categorical tensor of size (width, height), width and height being the cropped size of the target image tensor
    """

    assert (
        patch_overlap % 2 == 0
    ), f"Patch overlap argument must be a pair number. The one specified was {patch_overlap}."

    image_tensor = decode_image(file_path=target_image_path)

    # Cut the image into patches of size patch_size
    # & format the image patches to feed the model.predict function
    (
        main_patches_dataset,
        right_side_patches_dataset,
        down_side_patches_dataset,
    ) = build_predictions_dataset(
        target_image_tensor=image_tensor,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )

    # Build the model
    # & apply saved weights to the built model
    model = load_saved_model(
        checkpoint_dir_path=checkpoint_dir_path,
        n_classes=n_classes,
        input_shape=patch_size,
        batch_size=batch_size,
        encoder_kernel_size=encoder_kernel_size,
    )

    # Make predictions on the patches
    # output : array of shape (n_patches, patch_size, patch_size, n_classes)
    main_patch_classes_list = patches_predict(
        predictions_dataset=main_patches_dataset,
        model=model,
    )
    right_side_patch_classes_list = patches_predict(
        predictions_dataset=right_side_patches_dataset,
        model=model,
    )
    down_side_patch_classes_list = patches_predict(
        predictions_dataset=down_side_patches_dataset,
        model=model,
    )

    # Rebuild the image with the predictions patches
    # output tensor of size (intput_width_size - 2 * patch_overlap, input_height_size - 2 * patch_overlap)
    final_predictions_tensor = rebuild_predictions_with_overlap(
        target_image_path=target_image_path,
        main_patch_classes_list=main_patch_classes_list,
        right_side_patch_classes_list=right_side_patch_classes_list,
        down_side_patch_classes_list=down_side_patch_classes_list,
        image_tensor=image_tensor,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        misclassification_size=misclassification_size,
    )

    print(
        f"\nPredictions on {get_image_name_without_extension(target_image_path)} have been done."
    )

    return final_predictions_tensor


def build_predictions_dataset(
    target_image_tensor: tf.Tensor, patch_size: int, patch_overlap: int
) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
    """
    Build a dataset of patches to make predictions on each one of them.

    :param target_image_tensor: Image to make predictions on.
    :param patch_size: Size of the patch.
    :param patch_overlap: Number of pixels on which neighbors patches intersect each other.
    :return: A Dataset object with tensors of size (1, patch_size, patch_size, 3). Its length corresponds of the number of patches generated.
    """
    print("\nSlice the image into patches...")
    (
        main_patches_tensors_list,
        right_side_patches_tensors_list,
        down_side_patches_tensors_list,
    ) = extract_patches(
        image_tensor=target_image_tensor,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )
    prediction_dataset = tf.data.Dataset.from_tensor_slices(main_patches_tensors_list)
    prediction_dataset = prediction_dataset.batch(
        batch_size=1, drop_remainder=True
    )

    right_side_prediction_dataset = tf.data.Dataset.from_tensor_slices(
        right_side_patches_tensors_list
    )
    right_side_prediction_dataset = right_side_prediction_dataset.batch(
        batch_size=1, drop_remainder=True
    )

    down_side_prediction_dataset = tf.data.Dataset.from_tensor_slices(
        down_side_patches_tensors_list
    )
    down_side_prediction_dataset = down_side_prediction_dataset.batch(
        batch_size=1, drop_remainder=True
    )

    print(f"\n{len(prediction_dataset)} patches created successfully.")
    print(
        f"\n{len(right_side_prediction_dataset)} right side patches created successfully."
    )
    print(
        f"\n{len(down_side_prediction_dataset)} down side patches created successfully."
    )

    return (
        prediction_dataset,
        right_side_prediction_dataset,
        down_side_prediction_dataset,
    )


def extract_patches(
    image_tensor: tf.Tensor,
    patch_size: int,
    patch_overlap: int,
    with_four_channels: bool = False,
) -> [tf.Tensor]:
    """
    Split an image into smaller patches.
    Padding is by default implemented as "VALID", meaning that only patches which are fully
    contained in the input image are included.

    :param image_tensor: Path of the image we want to cut into patches.
    :param patch_size: Size of the patch.
    :param patch_overlap: Number of pixels on which neighbors patches intersect each other.
    :param with_four_channels: Set it to True if the image is a PNG. Default to False for JPEG.
    :return: A list of patches of the original image.
    """
    image_tensor = tf.expand_dims(image_tensor, 0)
    # if the image is a png, drop the brightness channel
    if with_four_channels:
        image_tensor = image_tensor[:, :, :, :3]

    image_height, image_width, channels_number = get_image_tensor_shape(
        image_tensor=image_tensor
    )
    window_stride = (
        patch_size - patch_overlap
    )  # number of pixels by which we shift the window at each step of predictions

    main_patches = list()
    right_side_patches = list()
    row_idx = 0
    while row_idx + patch_size <= image_height:
        column_idx = 0
        while column_idx + patch_size <= image_width:
            patch = image_tensor[
                :,
                row_idx : row_idx
                + patch_size,  # max bound  index is row_idx + patch_size - 1
                column_idx : column_idx
                + patch_size,  # max bound index is column_idx + patch_size - 1
                :,
            ]
            main_patches.append(patch[0])
            column_idx += window_stride

        # extract right side patches
        down_right_side_patch = image_tensor[
            :, row_idx : row_idx + patch_size, image_width - patch_size : image_width, :
        ]
        right_side_patches.append(down_right_side_patch[0])

        row_idx += window_stride

    # extract down side patches
    down_side_patches = list()
    column_idx = 0
    while column_idx + patch_size <= image_width:
        down_side_patch = image_tensor[
            :,
            image_height - patch_size : image_height,
            column_idx : column_idx + patch_size,
            :,
        ]
        down_side_patches.append(down_side_patch[0])
        column_idx += window_stride

    # down-right corner
    down_right_side_patch = image_tensor[
        :,
        image_height - patch_size : image_height,
        image_width - patch_size : image_width,
        :,
    ]
    right_side_patches.append(down_right_side_patch[0])

    n_vertical_patches = (image_height - 2 * int(patch_overlap / 2)) // window_stride
    n_horizontal_patches = (image_width - 2 * int(patch_overlap / 2)) // window_stride
    assert n_vertical_patches * n_horizontal_patches == len(
        main_patches
    ), f"The number of main patches is not the same : original image of size {image_height}x{image_width} should have {n_horizontal_patches * n_vertical_patches} but we have {len(main_patches)} "
    assert (n_vertical_patches + 1) == len(
        right_side_patches
    ), f"The number of right side patches is not the same : is {len(right_side_patches)}, should be {n_vertical_patches + 1}"
    assert n_horizontal_patches == len(
        down_side_patches
    ), f"The number of right side patches is not the same : is {len(down_side_patches)}, should be {n_horizontal_patches}"

    return main_patches, right_side_patches, down_side_patches


def patches_predict(
    predictions_dataset: tf.data.Dataset,
    model: tf.keras.Model,
) -> [tf.Tensor]:
    predictions: np.ndarray = model.predict(x=predictions_dataset, verbose=1)

    # Remove background predictions so it takes the max on the non background classes
    # Note : the argmax function shift the classes numbers of -1, that is why we add one just after
    patch_classes_list = list(np.argmax(predictions[:, :, :, 1:], axis=3))
    patch_classes_list = [
        tf.constant(
            np.add(patch_classes, np.ones(patch_classes.shape, dtype=np.int32)),
            dtype=tf.int32,
        )
        for patch_classes in patch_classes_list
    ]
    # Tensors classes list is of size n_patches of tensor with size (patch_size, patch_size)

    return patch_classes_list


def rebuild_predictions_with_overlap(
        target_image_path: Path,
        main_patch_classes_list: [tf.Tensor],
        right_side_patch_classes_list: [tf.Tensor],
        down_side_patch_classes_list: [tf.Tensor],
        image_tensor: tf.Tensor,
        patch_size: int,
        patch_overlap: int,
        misclassification_size: int = 5,
) -> tf.Tensor:
    """
    Restructure the patches that were generated with the extract_patches_with_overlap() function.
    Warning : This function is strongly coupled with the function extract_patches() from the patches_generator.py module

    :param target_image_path:
    :param down_side_patch_classes_list:
    :param main_patch_classes_list: List with size n_patches of tensor with size (patch_size, patch_size)
    :param right_side_patch_classes_list:
    :param image_tensor:
    :param patch_size: Size of the patches.
    :param patch_overlap: Number of pixels on which neighbors patches intersect each other.
    :param misclassification_size: Estimated number of pixels on which the classification is wrong due to side effects between neighbors patches.
    :return: The rebuilt image tensor with dimension : [width, height, 1].
    """
    assert (
            patch_overlap % 2 == 0
    ), f"Patch overlap argument must be a pair number. The one specified was {patch_overlap}."
    assert (
            misclassification_size <= patch_overlap / 2
    ), f"Please increase the patch overlap (currently {patch_overlap}) to at least 2 times the misclassification size (currently {misclassification_size})."

    # Turns the patches in the list to a size of (patch_size, patch_size, 1)
    main_patch_classes_list = [
        tf.expand_dims(input=patch_classes, axis=2)
        for patch_classes in main_patch_classes_list
    ]

    right_side_patch_classes_list = [
        tf.expand_dims(input=patch_classes, axis=2)
        for patch_classes in right_side_patch_classes_list
    ]

    down_side_patch_classes_list = [
        tf.expand_dims(input=patch_classes, axis=2)
        for patch_classes in down_side_patch_classes_list
    ]

    # Counting the number of main patches by which the image has been cut
    image_height, image_width, channels_number = get_image_tensor_shape(
        image_tensor=image_tensor
    )

    window_stride = (
            patch_size - patch_overlap
    )  # number of pixels by which we shift the window at each step of predictions
    n_vertical_patches = (image_height - 2 * int(patch_overlap / 2)) // window_stride
    n_horizontal_patches = (image_width - 2 * int(patch_overlap / 2)) // window_stride

    print(
        f"\nRebuilding predictions patches for image {get_file_name_with_extension(target_image_path)}..."
    )
    for row_number in range(n_vertical_patches):
        for column_number in range(n_horizontal_patches):
            # Rebuild the line
            patch_number = row_number * n_horizontal_patches + column_number
            patch_tensor = main_patch_classes_list[patch_number]

            # cropping the patch by taking into account the overlap with which it was built
            cropped_patch_tensor = crop_patch_tensor(
                patch_tensor=patch_tensor, patch_overlap=patch_overlap
            )
            if column_number == 0:
                line_rebuilt_tensor = cropped_patch_tensor
            else:
                line_rebuilt_tensor = tf.concat(
                    [line_rebuilt_tensor, cropped_patch_tensor], axis=1
                )

        # first add the right side patches to extend the right side
        right_side_patch_tensor = right_side_patch_classes_list[row_number]
        cropped_right_side_patch_tensor = crop_patch_tensor(
            patch_tensor=right_side_patch_tensor, patch_overlap=patch_overlap
        )
        left_bound_limit_idx = (
                                       n_horizontal_patches * window_stride + int(patch_overlap / 2)
                               ) - (image_width - patch_size + int(patch_overlap / 2))
        resized_cropped_right_side_patch_tensor = cropped_right_side_patch_tensor[
                                                  :, left_bound_limit_idx:
                                                  ]
        line_rebuilt_tensor = tf.concat(
            [line_rebuilt_tensor, resized_cropped_right_side_patch_tensor], axis=1
        )

        if row_number == 0:
            rebuilt_tensor = line_rebuilt_tensor
        else:
            rebuilt_tensor = tf.concat([rebuilt_tensor, line_rebuilt_tensor], axis=0)

    # finally add the down side patches to extend the image bottom
    for column_number in range(n_horizontal_patches):
        down_side_patch_tensor = down_side_patch_classes_list[column_number]

        # cropping the patch by taking into account the overlap with which it was built
        cropped_down_side_patch_tensor = crop_patch_tensor(
            patch_tensor=down_side_patch_tensor, patch_overlap=patch_overlap
        )
        up_bound_limit_idx = (
                                     n_vertical_patches * window_stride + int(patch_overlap / 2)
                             ) - (image_height - patch_size + int(patch_overlap / 2))
        resized_cropped_down_side_patch_tensor = cropped_down_side_patch_tensor[
                                                 up_bound_limit_idx:, :
                                                 ]

        if column_number == 0:
            line_rebuilt_tensor = resized_cropped_down_side_patch_tensor
        else:
            line_rebuilt_tensor = tf.concat(
                [line_rebuilt_tensor, resized_cropped_down_side_patch_tensor], axis=1
            )
    # add the right-down corner patch
    down_right_side_patch_tensor = right_side_patch_classes_list[n_vertical_patches]
    cropped_down_right_side_patch_tensor = crop_patch_tensor(
        patch_tensor=down_right_side_patch_tensor, patch_overlap=patch_overlap
    )
    left_bound_limit_idx = (
                                   n_horizontal_patches * window_stride + int(patch_overlap / 2)
                           ) - (image_width - patch_size + int(patch_overlap / 2))
    up_bound_limit_idx = (
                                 n_vertical_patches * window_stride + int(patch_overlap / 2)
                         ) - (image_height - patch_size + int(patch_overlap / 2))
    resized_cropped_down_right_side_patch_tensor = cropped_down_right_side_patch_tensor[
                                                   up_bound_limit_idx:, left_bound_limit_idx:
                                                   ]
    line_rebuilt_tensor = tf.concat(
        [line_rebuilt_tensor, resized_cropped_down_right_side_patch_tensor], axis=1
    )

    rebuilt_tensor = tf.concat([rebuilt_tensor, line_rebuilt_tensor], axis=0)

    # Check that the final size is consistent
    (
        rebuilt_tensor_height,
        rebuilt_tensor_width,
        rebuilt_channels_number,
    ) = get_image_tensor_shape(image_tensor=rebuilt_tensor)
    assert rebuilt_tensor_height == (
        int(image_height - 2 * (patch_overlap / 2))
    ), f"Number of rows is not consistent : got {rebuilt_tensor_height}, expected {int(image_height - 2 * (patch_overlap / 2))}"
    assert rebuilt_tensor_width == (
        int(image_width - 2 * (patch_overlap / 2))
    ), f"Number of columns is not consistent : got {rebuilt_tensor_width}, expected {int(image_width - 2 * (patch_overlap / 2))}"

    rebuilt_tensor = tf.squeeze(input=rebuilt_tensor)
    print(
        f"\nImage predictions have been successfully built with size {rebuilt_tensor.shape} (original image size : {image_tensor.shape})."
    )
    return rebuilt_tensor


def crop_patch_tensor(
    patch_tensor: tf.Tensor,
    patch_overlap: int,
) -> tf.Tensor:
    """

    :param patch_tensor: A tensor of size (x, y).
    :param patch_overlap: Number of pixels on which neighbors patches intersect each other.
    :return: A cropped tensor of size (x - patch_overlap, y - patch_overlap).
    """
    assert (
        patch_overlap % 2 == 0
    ), f"Patch overlap argument must be a pair number. The one specified was {patch_overlap}."

    image_height, image_width, channels_number = get_image_tensor_shape(
        image_tensor=patch_tensor
    )
    target_width = int(image_width - 2 * (patch_overlap / 2))
    target_height = int(image_height - 2 * (patch_overlap / 2))

    patch_tensor = crop_tensor(
        tensor=patch_tensor, target_height=target_height, target_width=target_width
    )
    return patch_tensor


def crop_tensor(tensor: tf.Tensor, target_height: int, target_width: int) -> tf.Tensor:
    """
    Returns a cropped tensor of size (target_height, target_width).
    First try to center the crop. If not possible, gives the most top-left part (see tests as example).

    :param tensor: The tensor to crop.
    :param target_height: Height of the output tensor.
    :param target_width: Width of the output tensor.
    :return: The cropped tensor of size (target_height, target_width).
    """
    tensor_height, tensor_width, channels_number = get_image_tensor_shape(
        image_tensor=tensor
    )

    # center the crop
    height_difference = tensor_height - target_height
    width_difference = tensor_width - target_width

    assert height_difference >= 0
    height_offset = int(height_difference // 2)

    assert width_difference >= 0
    width_offset = int(width_difference // 2)

    # cropping and saving
    cropped_tensor = tf.image.crop_to_bounding_box(
        image=tensor,
        offset_height=height_offset,
        offset_width=width_offset,
        target_height=target_height,
        target_width=target_width,
    )

    # Check that the output size is correct
    (
        cropped_tensor_height,
        cropped_tensor_width,
        channels_number,
    ) = get_image_tensor_shape(image_tensor=cropped_tensor)
    assert (cropped_tensor_height == target_height) and (
        cropped_tensor_height == target_width
    ), f"\nCropped tensor shape is ({cropped_tensor_height}, {cropped_tensor_width}) : should be {target_height}, {target_width}"

    return cropped_tensor
