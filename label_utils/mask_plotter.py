import matplotlib.pyplot as plt
import matplotlib.image as mpimg

mask_path


def plot_label_mask(mask_path: str) -> None:
    """Plot with matplotlib"""
    img = mpimg.imread(mask_path)
    plt.imshow(img)
    plt.show()
    return img
