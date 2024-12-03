import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from keras.models import load_model
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt


# Initialize Flask application
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load pre-trained face detector and emotion model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Generate a bar chart for emotion distribution
def create_emotion_chart(emotions):
    # Create a bar chart for detected emotions
    labels = [emotion[0] for emotion in emotions]
    values = [emotion[1] for emotion in emotions]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color='skyblue', alpha=0.8)
    plt.xlabel('Emotions')
    plt.ylabel('Confidence')
    plt.title('Emotion Confidence Distribution')
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'emotion_chart.png'))
    plt.close()

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Image upload and processing route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Read the uploaded image and process it
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        emotion_results = []

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            emotion_prediction = emotion_model.predict(face)
            emotion_label = emotion_labels[np.argmax(emotion_prediction)]
            emotion_accuracy = float(np.max(emotion_prediction))  # Convert to float for JSON serialization

            # Draw rectangle and label on the image
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, f"{emotion_label} ({emotion_accuracy:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            emotion_results.append((emotion_label, emotion_accuracy))

        # Save the processed image
        output_filename = 'output_' + filename
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, image)

        # Generate emotion chart
        create_emotion_chart(emotion_results)

        return render_template('display.html', filename=output_filename, emotions=emotion_results)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
