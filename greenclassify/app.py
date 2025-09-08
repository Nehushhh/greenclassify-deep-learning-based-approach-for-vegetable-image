from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join('model', 'greenclassify_cnn_model.h5')
model = load_model(MODEL_PATH)

# Upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Class labels
CLASSES = [
    'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
    'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
    'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
]

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# About route
@app.route('/about')
def about():
    return render_template('about.html')

# Contact route
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Predict route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    image_path = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image
            img = image.load_img(filepath, target_size=(128, 128))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            # Make prediction
            preds = model.predict(x)
            class_idx = np.argmax(preds)
            prediction = CLASSES[class_idx]

            # Path for displaying uploaded image
            image_path = url_for('static', filename=f'uploads/{filename}')

    return render_template('predict.html', prediction=prediction, image_path=image_path)

# Logout route
@app.route('/logout')
def logout():
    return render_template('logout.html')

if __name__ == '__main__':
    app.run(debug=True)
