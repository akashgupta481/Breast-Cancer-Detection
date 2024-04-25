from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained Breast Cancer Detection model
model_path = "/Users/akash/MCA Academics/Minor Project/Breast Cancer Detection Final/breast_cancer_model.h5"
model = load_model(model_path)

# Define function to preprocess image for Breast Cancer Detection
def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(48, 48))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize the image
    return img

# Define route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define route to handle image upload and Breast Cancer Detection prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Save the uploaded file temporarily
        file_path = 'temp_img.jpg'
        file.save(file_path)

        # Preprocess the image for Breast Cancer Detection
        processed_img = preprocess_image(file_path)

        # Make prediction
        result = model.predict(processed_img)
        prediction = 'Benign' if result[0][0] > 0.5 else 'Malignant'

        # Remove the temporary file
        os.remove(file_path)

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
