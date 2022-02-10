from pathlib import Path
from ui_integration.predictions_maker import make_predictions
from ui_integration.utils import get_formatted_time, get_image_name_without_extension, turn_2d_tensor_to_3d_tensor

import tensorflow as tf

# Constants
PATCH_SIZE = 256
PATCH_OVERLAP = 40
N_CLASSES = 9
BATCH_SIZE = 8
ENCODER_KERNEL_SIZE = 3
MAPPING_CLASS_NUMBER = {
    "background": 0,
    "poils-cheveux": 1,
    "vetements": 2,
    "peau": 3,
    "bois-tronc": 4,
    "ciel": 5,
    "feuilles-vertes": 6,
    "herbe": 7,
    "eau": 8,
    "roche": 9,
}  # Maps each labelling class to a number
# Values in a binary LabelBox mask
MASK_TRUE_VALUE = 255
MASK_FALSE_VALUE = 0


# todo : change report_dir_path with "path where to store binary classifications"
def main(image_path: Path, report_dir_path: Path):
    build_predict_run_report(
        image_path=image_path,
        report_dir_path=report_dir_path,
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP,
        n_classes=N_CLASSES,
        batch_size=BATCH_SIZE,
        encoder_kernel_size=ENCODER_KERNEL_SIZE,
    )


def build_predict_run_report(
    image_path: Path,
    report_dir_path: Path,
    patch_size: int,
    patch_overlap: int,
    n_classes: int,
    batch_size: int,
    encoder_kernel_size: int,
) -> None:
    predictions_report_root_path = (
        report_dir_path / "3_predictions" / get_formatted_time()
    )
    predictions_report_root_path.mkdir(parents=True)

    # Make predictions
    predictions_tensor = make_predictions(
        target_image_path=image_path,
        checkpoint_dir_path=report_dir_path / "2_model_report",
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        n_classes=n_classes,
        batch_size=batch_size,
        encoder_kernel_size=encoder_kernel_size,
    )

    # todo : ask if Alexandre wants to keep it ?
    # save_test_images_vs_predictions_plot(
    #     target_image_path=image_path,
    #     predictions_tensor=predictions_tensor,
    #     predictions_report_root_path=predictions_report_root_path,
    # )

    # todo : ask if Alexandre wants to keep it ?
    # predictions_only_path = save_predictions_only_plot(
    #     target_image_path=image_path,
    #     predictions_tensor=predictions_tensor,
    #     predictions_report_root_path=predictions_report_root_path,
    # )

    save_binary_predictions_plot(
        target_image_path=image_path,
        predictions_tensor=predictions_tensor,
        predictions_report_root_path=predictions_report_root_path,
    )


def save_binary_predictions_plot(
    target_image_path: Path,
    predictions_tensor: tf.Tensor,
    predictions_report_root_path: Path,
):
    # Separate predictions tensor into a list of n_classes binary tensors of size (width, height)
    binary_predictions_sub_dir = predictions_report_root_path / "binary_predictions"
    if not binary_predictions_sub_dir.exists():
        binary_predictions_sub_dir.mkdir()

    # Create and save binary tensors
    for idx, class_number in enumerate(MAPPING_CLASS_NUMBER.values()):
        binary_tensor = tf.where(
            condition=tf.equal(predictions_tensor, class_number),
            x=MASK_TRUE_VALUE,
            y=MASK_FALSE_VALUE,
        )

        binary_tensor_3d = turn_2d_tensor_to_3d_tensor(tensor_2d=binary_tensor)
        mapping_number_class = {
            class_number: class_name
            for class_name, class_number in MAPPING_CLASS_NUMBER.items()
        }

        binary_predictions_class_sub_dir = (
            predictions_report_root_path
            / "binary_predictions"
            / get_image_name_without_extension(target_image_path)
        )
        if not binary_predictions_class_sub_dir.exists():
            binary_predictions_class_sub_dir.mkdir(parents=True)
        output_path = (
            binary_predictions_sub_dir
            / get_image_name_without_extension(target_image_path)
            / f"{get_image_name_without_extension(target_image_path)}__{mapping_number_class[idx]}.png"
        )
        tf.keras.preprocessing.image.save_img(output_path, binary_tensor_3d)
        print(f"\nBinary predictions plot successfully saved at : {output_path}")
