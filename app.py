from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from Flutter app

# Load the trained ML model
model = pickle.load(open("models/mcb_trip_model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    temperature = data['temperature']
    humidity = data['humidity']
    water_detected = 1 if data['water_detected'].lower() == 'yes' else 0
    hour = data['hour']
    day = data['day']
    month = data['month']

    # Convert data to numpy array
    input_data = np.array([[temperature, humidity, water_detected, hour, day, month]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return jsonify({'prediction': "MCB Trip" if prediction == 1 else "No Trip"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
