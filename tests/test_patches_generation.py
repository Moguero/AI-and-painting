import tensorflow as tf

n = 10
test_image = tf.constant([[[[x * n + y + 1] * 3 for y in range(n)] for x in range(n)]])

patch_size = 2  # with n=10, creates 25 patches of size 2x2
patches = tf.image.extract_patches(
    images=test_image,
    sizes=[1, patch_size, patch_size, 1],
    strides=[1, patch_size, patch_size, 1],
    rates=[1, 1, 1, 1],
    padding='VALID'
)
reshaped_patches = tf.reshape(
    tensor=patches,
    shape=[
        patches.shape[1] * patches.shape[2],
        patch_size,
        patch_size,
        3
    ]
)
