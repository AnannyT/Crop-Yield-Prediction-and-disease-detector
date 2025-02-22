import pandas as pd

file_path = "crop_recommendation_with_variability.csv"
crop_data = pd.read_csv(file_path)


moderate_indian_yields = {
    'rice': 3500,  
    'maize': 2800,
    'chickpea': 800,
    'kidneybeans': 900,
    'pigeonpeas': 800,
    'mothbeans': 750,
    'mungbean': 700,
    'blackgram': 700,
    'lentil': 1000,
    'pomegranate': 10000,
    'banana': 30000,
    'mango': 12000,
    'grapes': 12000,
    'watermelon': 25000,
    'muskmelon': 20000,
    'apple': 15000,
    'orange': 15000,
    'papaya': 25000,
    'coconut': 10000,
    'cotton': 1200,
    'jute': 2500,
    'coffee': 2000
}


crop_optimal_conditions = {
    'rice': {'N': (80, 120), 'P': (30, 50), 'K': (40, 60), 'temp': (20, 35), 'rain': (150, 300), 'ph': (5.5, 7.5)},
    'maize': {'N': (100, 150), 'P': (50, 80), 'K': (50, 70), 'temp': (20, 30), 'rain': (200, 300), 'ph': (5.8, 7.0)},
    'chickpea': {'N': (40, 60), 'P': (20, 40), 'K': (20, 40), 'temp': (20, 30), 'rain': (300, 500), 'ph': (6.0, 7.5)},
    'kidneybeans': {'N': (40, 80), 'P': (30, 60), 'K': (30, 50), 'temp': (20, 30), 'rain': (300, 600), 'ph': (6.0, 7.5)},
    'pigeonpeas': {'N': (60, 80), 'P': (20, 50), 'K': (40, 60), 'temp': (25, 35), 'rain': (500, 800), 'ph': (6.0, 7.5)},
    'mothbeans': {'N': (40, 60), 'P': (30, 50), 'K': (30, 50), 'temp': (25, 35), 'rain': (400, 600), 'ph': (6.0, 7.5)},
    'mungbean': {'N': (50, 80), 'P': (30, 50), 'K': (30, 60), 'temp': (25, 35), 'rain': (400, 600), 'ph': (6.0, 7.0)},
    'blackgram': {'N': (50, 80), 'P': (30, 60), 'K': (30, 60), 'temp': (25, 35), 'rain': (300, 600), 'ph': (6.0, 7.5)},
    'lentil': {'N': (40, 70), 'P': (20, 40), 'K': (20, 40), 'temp': (15, 25), 'rain': (300, 500), 'ph': (6.0, 7.5)},
    'pomegranate': {'N': (80, 120), 'P': (40, 70), 'K': (60, 90), 'temp': (25, 40), 'rain': (300, 600), 'ph': (6.0, 7.0)},
    'banana': {'N': (200, 300), 'P': (80, 120), 'K': (150, 200), 'temp': (25, 35), 'rain': (1000, 1500), 'ph': (5.5, 7.0)},
    'mango': {'N': (100, 150), 'P': (40, 70), 'K': (60, 90), 'temp': (25, 35), 'rain': (800, 1200), 'ph': (5.5, 7.5)},
    'grapes': {'N': (100, 150), 'P': (50, 80), 'K': (60, 90), 'temp': (20, 30), 'rain': (500, 1000), 'ph': (6.0, 7.0)},
    'watermelon': {'N': (100, 150), 'P': (40, 60), 'K': (60, 80), 'temp': (25, 35), 'rain': (400, 600), 'ph': (6.0, 7.5)},
    'muskmelon': {'N': (80, 120), 'P': (40, 70), 'K': (50, 80), 'temp': (25, 35), 'rain': (400, 600), 'ph': (6.0, 7.5)},
    'apple': {'N': (100, 150), 'P': (50, 80), 'K': (60, 90), 'temp': (10, 25), 'rain': (500, 1000), 'ph': (6.0, 7.5)},
    'orange': {'N': (100, 150), 'P': (40, 70), 'K': (60, 90), 'temp': (20, 30), 'rain': (400, 800), 'ph': (5.5, 7.0)},
    'papaya': {'N': (200, 300), 'P': (50, 100), 'K': (100, 150), 'temp': (25, 35), 'rain': (1000, 1500), 'ph': (6.0, 7.5)},
    'coconut': {'N': (150, 250), 'P': (50, 100), 'K': (100, 150), 'temp': (25, 35), 'rain': (1500, 2000), 'ph': (5.5, 7.0)},
    'cotton': {'N': (100, 150), 'P': (40, 60), 'K': (40, 70), 'temp': (25, 35), 'rain': (150, 300), 'ph': (6.0, 8.0)},
    'jute': {'N': (80, 120), 'P': (30, 50), 'K': (40, 60), 'temp': (20, 30), 'rain': (1000, 1500), 'ph': (6.0, 7.5)},
    'coffee': {'N': (100, 150), 'P': (50, 80), 'K': (80, 100), 'temp': (15, 25), 'rain': (2000, 3000), 'ph': (5.5, 7.0)}
}


def calculate_yield_balanced_indian(row):
    crop = row['label']
    if crop not in crop_optimal_conditions or crop not in moderate_indian_yields:
        return None  

    
    optimal = crop_optimal_conditions[crop]
    base_yield = moderate_indian_yields[crop]

   
    def penalty(value, low, high, weight=1.2):
        if low <= value <= high:
            return 1
        deviation = min(abs(value - low), abs(value - high))
        range_width = high - low
        return max(0.5, 1 - weight * deviation / range_width)  # Moderate penalties for deviations

    
    nutrient_penalty = (penalty(row['N'], *optimal['N'], weight=1.1) +
                        penalty(row['P'], *optimal['P'], weight=1.0) +
                        penalty(row['K'], *optimal['K'], weight=0.9)) / 3
    temp_penalty = penalty(row['temperature'], *optimal['temp'], weight=0.9)
    rain_penalty = penalty(row['rainfall'], *optimal['rain'], weight=1.0)
    ph_penalty = penalty(row['ph'], *optimal['ph'], weight=0.8)

    
    total_penalty = nutrient_penalty * temp_penalty * rain_penalty * ph_penalty
    return base_yield * total_penalty  


crop_data['calculated_yield'] = crop_data.apply(calculate_yield_balanced_indian, axis=1)


output_file_path = "updated_crop_recommendation2.csv"
crop_data.to_csv(output_file_path, index=False)

print(f"Updated CSV file saved to {output_file_path}")
