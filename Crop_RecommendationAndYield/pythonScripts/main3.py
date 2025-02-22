import numpy as np
import pandas as pd
import pickle
import requests
import joblib
import datetime


model_path = r'D:\college\MInor project\RandomForest.pkl'
with open(model_path, 'rb') as model_file:
    RF = pickle.load(model_file)

# Fetch weather data (temperature and humidity)
def weather_fetch(city_name):
    api_key = "9d7cde1f6d07ec55650544be1631307e"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city_name}"
    response = requests.get(complete_url)
    data = response.json()

    if data["cod"] != "404":
        main_data = data["main"]
        temperature = round(main_data["temp"] - 273.15, 2)  # Convert from Kelvin to Celsius
        humidity = main_data["humidity"]
        return temperature, humidity
    else:
        print("City not found.")
        return None, None

# Fetch historical rainfall data
import requests
import datetime

def fetch_rainfall(city_name):
    
    api_key = 'UENM8BX2959PUSFYWHV5CURHH'
    
    
    base_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city_name}"

    
    current_year = datetime.datetime.now().year
    start_date = f"{current_year - 1}-01-01"  
    end_date = f"{current_year - 1}-12-31"    

    
    url = f"{base_url}/{start_date}/{end_date}?key={api_key}&include=days"

    
    response = requests.get(url)

    if response.status_code == 200:
        try:
            
            print("Response Data:", response.json())
            
            data = response.json()
            total_rainfall = 0
            for day in data['days']:
                if 'precipitation' in day:
                    total_rainfall += day['precipitation']
            
            return total_rainfall
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return 0
    else:
        print(f"Error fetching data: {response.status_code}")
        print(f"Response Content: {response.text}")
        return 0

    
def recommend_crop():
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    if input_data is not None:
        prediction = RF.predict(input_data)
        print(f"Recommended crop: {prediction[0]}")
        return prediction[0]
    return 0


def yield_predict(crop):
    pipeline = joblib.load('crop_yield_model.pkl')
    user_data = pd.DataFrame({
        'Crop': [crop],
        'Temperature': [temperature],
        'Rainfall': [rainfall * 10],  
        'Humidity': [humidity],
        'Soil pH': [ph]
    })
    predicted_yield = pipeline.predict(user_data)
    print(f"Predicted Crop Yield: {round(predicted_yield[0], 2)} kg/ha")

if __name__ == "__main__":

    # User inputs for crop recommendation
    print("Please enter the following details for crop recommendation:")
    N = float(input("Nitrogen (N) content in soil: "))
    P = float(input("Phosphorus (P) content in soil: "))
    K = float(input("Potassium (K) content in soil: "))
    ph = float(input("pH value of soil: "))
    
    city = input("Enter city where farm located: ")

    
    temperature, humidity = weather_fetch(city)

  
    rainfall = fetch_rainfall(city)

    if temperature is None or humidity is None:
        print("Error fetching weather data. Please try again.")
    else:
        crop = recommend_crop()
        yield_predict(crop)
        print(f"rainfall is {rainfall}")
