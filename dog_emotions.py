import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the input shape of the images
input_shape = (224, 224, 3)

# Create a CNN model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Create an ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

# Load the training data from a directory
train_data = train_datagen.flow_from_directory('C:/Users/Aayush Srivastava/New folder/dog_emot/images',
                                               target_size=input_shape[:2],
                                               batch_size=32,
                                               class_mode='categorical')

# Create an ImageDataGenerator for preprocessing the validation data
val_datagen = ImageDataGenerator(rescale=1./255)

# Load the validation data from a directory
val_data = val_datagen.flow_from_directory('C:/Users/Aayush Srivastava/New folder/dog_emot/images',
                                            target_size=input_shape[:2],
                                            batch_size=32,
                                            class_mode='categorical')

# Train the model
model.fit(train_data,
          validation_data=val_data,
          epochs=10)

# Save the model
model.save('dog_emotions_model.h5')
