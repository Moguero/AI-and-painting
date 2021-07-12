import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_label_mask(mask_path: str) -> None:
    """Plot with matplotlib"""
    img = mpimg.imread(mask_path)
    plt.imshow(img)
    plt.show()
    return img


# todo : function to plot both the image and its masks with transparency on the masks


# todo : plot all the masks on the same image, with different colors

# delete the alpha channel not to have transparent channel

