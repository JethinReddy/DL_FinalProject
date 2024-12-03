import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
label_to_index = {label: idx for idx, label in enumerate(emotion_labels)}

# Load and preprocess dataset
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in emotion_labels:
        label_folder = os.path.join(folder, label)
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(label_to_index[label])
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Load training data
train_dir = 'C:\\Users\\jethi\\Downloads\\archive (6)\\train'
  # Replace with your dataset's training folder
X, y = load_images_from_folder(train_dir)

# Preprocess data
X = X / 255.0
X = np.expand_dims(X, -1)  # Add channel dimension for grayscale
y = to_categorical(y, num_classes=len(emotion_labels))

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(emotion_labels), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=40, batch_size=64, validation_data=(X_val, y_val))

# Save the model
model.save('emotion_model.h5')
