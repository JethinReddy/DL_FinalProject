import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
label_to_index = {label: idx for idx, label in enumerate(emotion_labels)}

# Function to load validation or test data
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

# Load validation or test data
val_dir = 'C:\\Users\\jethi\\Downloads\\archive (6)\\test'  # Path to your test/validation folder
X_val, y_val = load_images_from_folder(val_dir)

# Preprocess the data
X_val = X_val / 255.0  # Normalize
X_val = np.expand_dims(X_val, -1)  # Add channel dimension for grayscale
y_val = to_categorical(y_val, num_classes=len(emotion_labels))  # Convert to one-hot encoding

# Load the trained model
model = load_model('emotion_model.h5')  # Ensure the path is correct

# Evaluate the model
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class indices
y_true = np.argmax(y_val, axis=1)           # Convert one-hot encoded labels to class indices

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=emotion_labels))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
