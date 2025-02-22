import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib


df = pd.read_csv("updated_crop_recommendation2.csv")  

X = df.drop("Yield", axis=1)  
y = df["Yield"]


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Crop']),  
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  
            ('scaler', StandardScaler())  
        ]), ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])  
    ]
)

model = RandomForestRegressor(
    n_estimators=200,       
    max_depth=15,           
    min_samples_split=10,   
    min_samples_leaf=5,     
    random_state=42         
)


pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
print(f"Cross-Validation R2 Scores: {cv_scores}")
print(f"Average R2 Score: {np.mean(cv_scores)}")


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation on Test Data:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.4f}")
print(f"Average Yield: {y.mean():.2f}")

joblib.dump(pipeline, "crop_yield_model2.pkl")
print("\nModel saved as crop_yield_model.pkl")
