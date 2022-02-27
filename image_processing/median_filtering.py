import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.image_utils import get_image_name_without_extension


def median_filter(image: np.ndarray, kernel_filter_size: int) -> np.ndarray:
    assert (
        kernel_filter_size % 2 == 1
    ), f"Kernel size for median filtering must be odd but {kernel_filter_size} was given."

    median_filtered_image = cv2.medianBlur(src=image, ksize=kernel_filter_size)
    return median_filtered_image


def save_median_filtering_comparison(
    source_image_path: Path, predictions_report_root_path: Path
) -> None:
    image = cv2.cvtColor(
        src=cv2.imread(filename=str(source_image_path)), code=cv2.COLOR_BGR2RGB
    )
    median_3 = median_filter(image=image, kernel_filter_size=3)
    median_5 = median_filter(image=image, kernel_filter_size=5)
    median_7 = median_filter(image=image, kernel_filter_size=7)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle(
        f"Image : {get_image_name_without_extension(image_path=source_image_path)}",
        fontsize="small",
    )
    ax1.set_title("Original")
    ax2.set_title("Blurred x3")
    ax3.set_title("Blurred x5")
    ax4.set_title("Blurred x7")
    ax1.imshow(image)
    ax2.imshow(median_3)
    ax3.imshow(median_5)
    ax4.imshow(median_7)
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")

    # Save the plot
    median_filtered_predictions_dir_path = (
        predictions_report_root_path / "median_filtering"
    )
    if not median_filtered_predictions_dir_path.exists():
        median_filtered_predictions_dir_path.mkdir()

    output_path = (
        median_filtered_predictions_dir_path
        / f"median_filtered_comparison__{get_image_name_without_extension(source_image_path)}.jpg"
    )

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
