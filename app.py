from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sklearn
import os
import pickle
import warnings

app = Flask(__name__)

loaded_model = pickle.load(open("model.pkl", 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = loaded_model.predict(single_pred)

    # def validate_input_values(N, P, K, temperature, humidity, ph, rainfall):
    #     # Define sensible ranges for input parameters
    #     N_min, N_max = 0, 140  # Nitrogen content in the soil (kg/ha)
    #     P_min, P_max = 5, 145  # Phosphorus content in the soil (kg/ha)
    #     K_min, K_max = 5, 205  # Potassium content in the soil (kg/ha)
    #     temperature_min, temperature_max = 7, 44  # Temperature in Celsius
    #     humidity_min, humidity_max = 13, 100  # Humidity percentage (0% to 100%)
    #     ph_min, ph_max = 2, 10  # pH level (0 to 14)
    #     rainfall_min, rainfall_max = 19, 300  # Rainfall in millimeters (mm)
    #
    #     if (N_min <= N <= N_max and P_min <= P <= P_max and K_min <= K <= K_max and
    #             temperature_min <= temperature <= temperature_max and
    #             humidity_min <= humidity <= humidity_max and
    #             ph_min <= ph <= ph_max and rainfall_min <= rainfall <= rainfall_max):
    #         return True
    #     else:
    #         return False

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    # # Check if input values are within the defined ranges
    # if validate_input_values(N, P, K, temperature, humidity, ph, rainfall):
    #     # If all input values are within the valid ranges, proceed with prediction
    #     pred = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
    #     # Rest of the code to print crop recommendations based on 'pred'
    #     best_crop = crop_mapping[pred]
    #     print(f"{best_crop} is the best crop to be cultivated right there")
    #
    # else:
    #     print("Sorry, we could not determine the best crop to be cultivated with the provided data.")

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return render_template('home.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)