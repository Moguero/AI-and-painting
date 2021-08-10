import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from deep_learning.models.unet import build_unet_2
from dataset_utils.dataset_builder import get_small_dataset, get_small_dataset_2
from pathlib import Path

IMAGE_PATH = Path("C:/Users:thiba:PycharmProjects:mission_IA_JCS:files:images:_DSC0043:_DSC0043.JPG")
MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks")
CATEGORICAL_MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/categorical_masks")
IMAGE_PATHS = [
    Path(
        "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/1/1.jpg"
    ),
    Path(
        "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0130/_DSC0130.jpg"
    ),
]
MASK_PATHS = [
    Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/_DSC0043/feuilles_vertes/mask__DSC0043_feuilles_vertes__3466c2cda646448fbe8f4927f918e247.png"),
    Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/_DSC0061/feuilles_vertes/mask__DSC0061_feuilles_vertes__eef687829eb641c59f63ad80199b0de0.png")
]
PATCHES_DIR = Path(r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\files\patches")
IMAGE_TYPE = 'JPG'
BATCH_SIZE = 2
TEST_PROPORTION = 0.2


N_CLASSES = 9
N_PATCHES = 10
INPUT_SHAPE = 256
# OPTIMIZER = "rmsprop"
OPTIMIZER = Adam(lr=1e-4)
LOSS_FUNCTION = "categorical_crossentropy"


def main():
    # Define the model
    model = build_unet_2(N_CLASSES, INPUT_SHAPE, BATCH_SIZE)

    # Compile the model
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)
    # model.compile(optimizer=Adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Init the callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint("my_checkpoint.h5", save_best_only=True)
    ]

    # Data init
    # X_train, X_test, y_train, y_test = get_small_dataset(PATCHES_DIR, N_PATCHES, N_CLASSES, BATCH_SIZE)
    dataset = get_small_dataset_2(PATCHES_DIR, N_PATCHES, N_CLASSES, BATCH_SIZE, TEST_PROPORTION)
    # todo : generate a test dataset and a train dataset in the get_dataset function

    # Fit the model
    epochs = 15
    breakpoint()
    # history = model.fit(x=X_train, y=y_train, epochs=epochs, callbacks=callbacks)
    history = model.fit(dataset, epochs=epochs, callbacks=callbacks)
    # todo : use model.fit but with a generator instead of a Dataset

    # Save the model weights in HDF5 format
    # model.save_weights('./saved_weights.h5')

    # Evaluate the model
    loss = model.evaluate(X_test, y_test, verbose=1)

    # Make a prediction
    predictions = model.predict(X_test)
    classes = np.argmax(predictions, axis=3)
    breakpoint()
