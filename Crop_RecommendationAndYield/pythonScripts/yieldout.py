import joblib
import pandas as pd
import numpy as np


pipeline = joblib.load('crop_yield_model2.pkl')

def get_user_input():
    print("Please enter the following details:")
    
    
    N = float(input("Enter the nitrogen content (N) in soil (kg/ha): "))
    P = float(input("Enter the phosphorus content (P) in soil (kg/ha): "))
    K = float(input("Enter the potassium content (K) in soil (kg/ha): "))
    temperature = float(input("Enter the average temperature (°C): "))
    humidity = float(input("Enter the humidity (%): "))
    ph = float(input("Enter the soil pH: "))
    rainfall = float(input("Enter the average rainfall (cm): "))
    crop_label = input("Enter the crop label (e.g., Wheat, Rice, Corn): ")
    
    
    user_input = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall],
        'Crop': [crop_label]
    })
    
    return user_input


user_data = get_user_input()


predicted_yield = pipeline.predict(user_data)


print(f"\nPredicted Crop Yield: {predicted_yield[0]:.2f} kg/ha")
