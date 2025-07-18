from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Make sure the path is correct (absolute or relative to this file)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)