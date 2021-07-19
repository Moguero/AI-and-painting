from tensorflow import keras
from deep_learning.models.unet import build_unet
from image_utils.image_decoder import get_dataset

IMAGE_PATH = "C:/Users:thiba:PycharmProjects:mission_IA_JCS:files:images:_DSC0043:_DSC0043.JPG"
MASKS_DIR = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks"
IMAGE_PATHS = [
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043/_DSC0043.JPG",
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0061/_DSC0061.JPG"
]
MASK_PATHS = [
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/_DSC0043/feuilles_vertes/mask__DSC0043_feuilles_vertes__3466c2cda646448fbe8f4927f918e247.png",
    "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/_DSC0061/feuilles_vertes/mask__DSC0061_feuilles_vertes__eef687829eb641c59f63ad80199b0de0.png"
]
IMAGE_TYPE = 'JPG'
BATCH_SIZE = 1


N_CLASSES = 9
OPTIMIZER = "rmsprop"
LOSS_FUNCTION = "sparse_categorical_crossentropy"

model = build_unet(N_CLASSES, BATCH_SIZE)

model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)

callbacks = [
    keras.callbacks.ModelCheckpoint("my_checkpoint.h5", save_best_only=True)
]

# Data init

(X_train, y_train), (X_test, y_test) = get_dataset(IMAGE_PATHS, MASK_PATHS, IMAGE_TYPE, BATCH_SIZE)


# Train the model, doing validation at the end of each epoch.

epochs = 15
model.fit(x=X_train, y=y_train, epochs=epochs, callbacks=callbacks)
