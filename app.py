from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load your model and scaler
model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    features = np.array([
        data['age'],
        data['restingBP'],
        data['serumcholestrol'],
        data['fastingbloodsugar'],
        data['restingrelectro'],
        data['noofmajorvessels']
    ]).reshape(1, -1)

    features_scaled = scaler.transform(features)
    prediction = int(model.predict(features_scaled)[0])

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run server on your local IP
