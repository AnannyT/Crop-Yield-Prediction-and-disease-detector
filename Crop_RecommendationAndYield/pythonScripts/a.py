from flask import Flask, request, jsonify, render_template
from main2 import weather_fetch, recommend_crop, yield_predict
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import load_img, img_to_array
import io
import numpy as np
import os
import keras




app = Flask(_name_)
CORS(app)
@app.route("/")
def index():
    return render_template("crop.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        N = data['nitrogen']
        P = data['phosphorous']
        K = data['potassium']
        ph = data['ph']
        rainfall = data['rainfall']
        city = data['city']

        # Fetch weather data for the city
        temperature, humidity = weather_fetch(city)
        if temperature is None or humidity is None:
            return jsonify({'error': 'Weather data not available for the given city.'}), 400

        # Get crop recommendation
        crop = recommend_crop(N, P, K, temperature, humidity, ph, rainfall)

        # Get yield prediction
        predicted_yield = yield_predict(crop, temperature, rainfall, humidity, ph)

        return jsonify({'crop': crop, 'predicted_yield': predicted_yield})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500



MODEL_PATH = '/Users/adarshamit1001/MInor project/Plant-Disease-Detection-Main/test_model/trained_plant_disease_model_1.keras'
model = load_model(MODEL_PATH)

validation_set = tf.keras.utils.image_dataset_from_directory(
    '/Users/adarshamit1001/MInor project/Plant-Disease-Detection-Main/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)
class_name = validation_set.class_names


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Convert file to byte stream and load the image
        img_stream = io.BytesIO(file.read())  # Convert the file to a BytesIO object
        img = load_img(img_stream, target_size=(128, 128))  # Load and resize the image
        img_array = img_to_array(img)  # Convert the image to a NumPy array
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Scale pixel values to [0, 1]
        
        # Predict the disease using the loaded model
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions)  # Get the index of the highest probability
        disease = class_name[class_idx]  # Map index to class label
        
        return jsonify({'disease': disease}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if _name_ == "_main_":
    app.run(debug=True)