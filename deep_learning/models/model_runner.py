from deep_learning.models.unet import build_unet
from tensorflow.keras.optimizers import SGD, Adam

N_CLASSES = 9
EPOCHS_NUMBER = 5

# Define the model
model = build_unet(N_CLASSES)

# Compile the model

model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=Adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs=100, batch_size=32)

# Save the model weights in HDF5 format
model.save_weights('saved_weights.h5')

# Evaluate the model
loss = model.evaluate(X, y, verbose=0)

# Make a prediction
predictions = model.predict(X)
