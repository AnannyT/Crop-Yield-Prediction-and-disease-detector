from flask import Flask, request, jsonify, render_template
from main2 import weather_fetch, recommend_crop, yield_predict
from flask_cors import CORS

app = Flask(__name__)
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

        if 'city' in data:  # State & City mode
            city = data['city']
            temperature, humidity = weather_fetch(city)
            if temperature is None or humidity is None:
                return jsonify({'error': 'Weather data not available for the given city.'}), 400
        elif 'temperature' in data and 'humidity' in data:  # Temp & Humidity mode
            temperature = data['temperature']
            humidity = data['humidity']
        else:
            return jsonify({'error': 'Invalid input data.'}), 400

        # Get crop recommendation
        crop = recommend_crop(N, P, K, temperature, humidity, ph, rainfall)

        # Get yield prediction
        predicted_yield = yield_predict(N, P, K, crop, temperature, rainfall, humidity, ph)

        return jsonify({'crop': crop, 'predicted_yield': predicted_yield})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
