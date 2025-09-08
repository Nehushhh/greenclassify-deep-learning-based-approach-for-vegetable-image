import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model('model/greenclassify_cnn_model.h5')

# CLASSES = [
#     'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
#     'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
#     'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
# ]
CLASSES: ['Bean', 'Bitter_Gourd', 'Brinjal']

# Test images
test_images = [
    "static/uploads/broccoli.jpg",
    "static/uploads/bean.jpg",
    "static/uploads/carrot.jpg",
    "static/uploads/bitterGourd.jpg",
    "static/uploads/bottlegourd.jpg",
    "static/uploads/brinjal.jpg"
]

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"⚠️ File not found: {img_path}")
        continue

    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = model.predict(x)
    print(f"{img_path} → {CLASSES[np.argmax(preds)]} | Raw: {preds}")
