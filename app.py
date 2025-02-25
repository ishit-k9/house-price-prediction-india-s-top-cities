from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
import json

app = Flask(__name__)

# Load the trained pipeline
with open("model_out_log.pkl", "rb") as file:
    pipe = pickle.load(file)

# Load location average price data
with open('location_avg_price.json', 'r') as f:
    location_avg_price = json.load(f)

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/predict_bangalore', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            # Get user inputs
            user_inputs = {
                'area_type': request.form['area_type'],
                'location': request.form['location'],
                'total_sqft': float(request.form['total_sqft']),
                'BHK': int(request.form['BHK']),
                'bath': int(request.form['bath']),
                'balcony': int(request.form.get('balcony', 1)),
                'availability': request.form['availability'],
                'price_cat': request.form['price_cat']
            }

            # Calculate derived features
            user_inputs['room_density'] = user_inputs['total_sqft'] / user_inputs['BHK']
            user_inputs['avg_price_per_sqft_loc'] = location_avg_price.get(
                user_inputs['location'], np.nan)

            # Create input DataFrame
            input_data = pd.DataFrame([user_inputs])
            
            # Make prediction
            prediction = pipe.predict(input_data)[0]
            
            return jsonify({
                'status': 'success',
                'prediction': f"₹{round(prediction, 2):,} Lakh"
            })

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            })

if __name__ == '__main__':
    app.run(debug=True)