import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
# Data preparation
train_path = "chest_xray/train"
val_path = "chest_xray/val"
test_path = "chest_xray/test"

img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(train_path, target_size=img_size, class_mode='binary', batch_size=batch_size)
val_gen = datagen.flow_from_directory(val_path, target_size=img_size, class_mode='binary', batch_size=batch_size)
test_gen = datagen.flow_from_directory(test_path, target_size=img_size, class_mode='binary', batch_size=batch_size, shuffle=False)

# CNN model
base_model = VGG16(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
x = Flatten()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

# Training
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save model
model.save("cnn_xray_model.h5")

# Evaluation
loss, acc = model.evaluate(test_gen)
print(f"Test Accuracy: {acc:.4f}")
