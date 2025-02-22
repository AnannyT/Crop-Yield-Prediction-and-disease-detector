import pandas as pd
import numpy as np


file_path = "crop_recommendation.csv"  
crop_data = pd.read_csv(file_path)


def introduce_variability(value, lower_limit, upper_limit, perturbation_range):
    new_value = value + np.random.uniform(-perturbation_range, perturbation_range)
    return max(lower_limit, min(upper_limit, new_value))  


crop_data['temperature'] = crop_data['temperature'].apply(
    lambda temp: introduce_variability(temp, lower_limit=-10, upper_limit=50, perturbation_range=5)
)
crop_data['humidity'] = crop_data['humidity'].apply(
    lambda hum: introduce_variability(hum, lower_limit=0, upper_limit=100, perturbation_range=10)
)
crop_data['rainfall'] = crop_data['rainfall'].apply(
    lambda rain: introduce_variability(rain, lower_limit=0, upper_limit=1000, perturbation_range=20)
)


output_file_path = "crop_recommendation_with_variability.csv"
crop_data.to_csv(output_file_path, index=False)

print(f"Updated dataset with variability saved to {output_file_path}")
