from tensorflow import keras
from deep_learning.models.unet import build_unet
from dataset_utils.dataset_builder import get_dataset
from pathlib import Path

IMAGE_PATH = Path("C:/Users:thiba:PycharmProjects:mission_IA_JCS:files:images:_DSC0043:_DSC0043.JPG")
MASKS_DIR = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks")
IMAGE_PATHS = [
    Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0043/_DSC0043.JPG"),
    Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/_DSC0061/_DSC0061.JPG")
]
MASK_PATHS = [
    Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/_DSC0043/feuilles_vertes/mask__DSC0043_feuilles_vertes__3466c2cda646448fbe8f4927f918e247.png"),
    Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/_DSC0061/feuilles_vertes/mask__DSC0061_feuilles_vertes__eef687829eb641c59f63ad80199b0de0.png")
]
IMAGE_TYPE = 'JPG'
BATCH_SIZE = 1


N_CLASSES = 9
OPTIMIZER = "rmsprop"
LOSS_FUNCTION = "sparse_categorical_crossentropy"

# Define the model
model = build_unet(N_CLASSES, BATCH_SIZE)


# Compile the model
model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)
# model.compile(optimizer=Adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Init the callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint("my_checkpoint.h5", save_best_only=True)
]

# Data init
(X_train, y_train), (X_test, y_test) = get_dataset(IMAGE_PATHS, MASK_PATHS, IMAGE_TYPE, BATCH_SIZE)


# Fit the model
epochs = 15
model.fit(x=X_train, y=y_train, epochs=epochs, callbacks=callbacks)

# Save the model weights in HDF5 format
model.save_weights('saved_weights.h5')

# Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=0)

# Make a prediction
predictions = model.predict(X_test)
