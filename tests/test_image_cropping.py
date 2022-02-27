import tensorflow as tf
import numpy as np

from image_processing.cropping import crop_tensor


def test_crop_tensor():
    tensor1 = tf.constant([[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]])
    tensor1 = tf.expand_dims(tensor1, axis=2)
    cropped_tensor1 = crop_tensor(tensor=tensor1, target_width=2, target_height=3)[
        :, :, 0
    ]

    tensor2 = tf.constant(
        [
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    tensor2 = tf.expand_dims(tensor2, axis=2)
    cropped_tensor2 = crop_tensor(tensor=tensor2, target_width=3, target_height=3)[
        :, :, 0
    ]

    tensor3 = tf.constant(
        [
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 3, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    tensor3 = tf.expand_dims(tensor3, axis=2)
    cropped_tensor3 = crop_tensor(tensor=tensor3, target_width=2, target_height=2)[
        :, :, 0
    ]

    assert np.array_equal(cropped_tensor1.numpy(), np.array([[1, 1], [2, 2], [2, 2]]))
    assert np.array_equal(
        cropped_tensor2.numpy(), np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
    )
    assert np.array_equal(cropped_tensor3.numpy(), np.array([[2, 2], [2, 3]]))
