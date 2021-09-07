import tensorflow as tf
import matplotlib.pyplot as plt
from constants import *


def plot_augmented_images(
    train_dataset: tf.data.Dataset, n_samples_per_image: int, batch_size: int
):
    """
    Plot augmented images.

    :param train_dataset: Dataset of images.
    :param n_images: Number of images in the dataset.
    :param n_samples_per_image:
    :param patch_size:
    :return:
    """
    n_images = len(train_dataset) * batch_size
    fig, axes = plt.subplots(n_images * n_samples_per_image, 1)
    for idx1, (images_batch, labels_batch) in enumerate(train_dataset):
        for idx2, image in enumerate(images_batch):
            ax = axes[idx1 * batch_size + idx2]
            ax.axis("off")
            ax.imshow(image)

    plt.show()


# plot_augmented_images(TRAIN_DATASET, 4, 2)


def flip_augmentation(tensor: tf.Tensor) -> tf.Tensor:
    """
    Flip augmentation.

    :param tensor:
    :return:
    """
    tensor = tf.image.random_flip_left_right(tensor)
    return tensor


def color_augmentation(tensor: tf.Tensor) -> tf.Tensor:
    """
    Color augmentation.

    :param tensor:
    :return:
    """
    tensor = tf.image.random_hue(tensor, 0.08)
    tensor = tf.image.random_saturation(tensor, 0.6, 1.6)
    tensor = tf.image.random_brightness(tensor, 0.05)
    tensor = tf.image.random_contrast(tensor, 0.7, 1.3)
    return tensor


# todo : implement zooming augmentation


# todo : advised to set num_parallel_calls to the number of cpus of the machine
# debug
def f():
    augmentations = [flip_augmentation, color_augmentation]
    for f in augmentations:
        dataset = TRAIN_DATASET.map(lambda x, y: (f(x), y), num_parallel_calls=4)
    return dataset
