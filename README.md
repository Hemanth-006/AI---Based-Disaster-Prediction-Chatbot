# AI---Based-Disaster-Prediction-Chatbot

#install these with pip
pip install scikit-learn pandas numpy

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

# --------------------------
# Generate synthetic dataset
# --------------------------
def generate_data():
    np.random.seed(0)
    data_size = 500

    temperature = np.random.normal(30, 5, data_size)        # Celsius
    humidity = np.random.normal(70, 10, data_size)          # Percentage
    rainfall = np.random.normal(50, 20, data_size)          # mm
    wind_speed = np.random.normal(20, 5, data_size)         # km/h
    seismic_activity = np.random.uniform(0, 10, data_size)  # Richter scale-like

    # Disaster labels: 0 - None, 1 - Flood, 2 - Wildfire, 3 - Earthquake
    labels = []
    for i in range(data_size):
        if rainfall[i] > 80 and humidity[i] > 75:
            labels.append(1)  # Flood
        elif temperature[i] > 35 and humidity[i] < 40:
            labels.append(2)  # Wildfire
        elif seismic_activity[i] > 7.0:
            labels.append(3)  # Earthquake
        else:
            labels.append(0)  # None

    df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'rainfall': rainfall,
        'wind_speed': wind_speed,
        'seismic_activity': seismic_activity,
        'disaster': labels
    })

    return df

# --------------------------
# Train model
# --------------------------
def train_model(df):
    X = df.drop('disaster', axis=1)
    y = df['disaster']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    return clf

# --------------------------
# Disaster labels
# --------------------------
disaster_labels = {
    0: "No Disaster Detected",
    1: "Flood Risk Detected",
    2: "Wildfire Risk Detected",
    3: "Earthquake Risk Detected"
}

# --------------------------
# Chatbot Interaction
# --------------------------
def chatbot_loop(model):
    print("Welcome to the AI Disaster Prediction Chatbot!")
    print("Type 'exit' anytime to quit.\n")

    while True:
        try:
            user_input = input("You: Enter location (e.g., Delhi or type 'exit'): ").strip()
            if user_input.lower() == 'exit':
                print("Bot: Stay safe! Goodbye.")
                break

            # Simulate sensor inputs or ask user
            temp = float(input("Temperature (°C): "))
            humid = float(input("Humidity (%): "))
            rain = float(input("Rainfall (mm): "))
            wind = float(input("Wind Speed (km/h): "))
            seismic = float(input("Seismic Activity (0-10): "))

            input_data = np.array([[temp, humid, rain, wind, seismic]])
            prediction = model.predict(input_data)[0]
            print(f"\nBot: {disaster_labels[prediction]}")

            if prediction == 1:
                print("Advice: Avoid low-lying areas and stay updated on weather forecasts.\n")
            elif prediction == 2:
                print("Advice: Avoid dry forest areas and report fires immediately.\n")
            elif prediction == 3:
                print("Advice: Follow earthquake safety protocols. Evacuate if advised.\n")
            else:
                print("Advice: All conditions look normal. No immediate risk detected.\n")

        except ValueError:
            print("Bot: Please enter valid numeric values for environmental inputs.\n")

# --------------------------
# Run the project
# --------------------------
if __name__ == "__main__":
    df = generate_data()
    model = train_model(df)
    chatbot_loop(model)
