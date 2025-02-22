import numpy as np
import pandas as pd
import pickle
import requests
import joblib


model_path = r'D:\college\Minor_Project\Crop_RecommendationAndYield\test_models\RandomForest.pkl'
with open(model_path, 'rb') as model_file:
    RF = pickle.load(model_file)


def weather_fetch(city_name):
    
    api_key = "9d7cde1f6d07ec55650544be1631307e"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city_name}"
    response = requests.get(complete_url)
    data = response.json()

    if data["cod"] != "404":
        main_data = data["main"]
        temperature = round(main_data["temp"] - 273.15, 2)  
        humidity = main_data["humidity"]
        return temperature, humidity
    else:
        print("City not found.")
        return None, None


def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    if input_data is not None:  
        prediction = RF.predict(input_data)
        print(f"Recommended crop: {prediction[0]}")
        return prediction[0]
    return 0

def yield_predict(N,P,K,crop, temperature, rainfall, humidity, ph):
    pipeline = joblib.load(r'D:\college\Minor_Project\Crop_RecommendationAndYield\test_models\crop_yield_model2.pkl')
    user_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall],
        'Crop': [crop]
    })
    predicted_yield = pipeline.predict(user_data)
    print(f"Recommended crop: {predicted_yield[0]}")
    return round(predicted_yield[0],2)


