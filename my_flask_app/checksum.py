from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('linear_regression_model.pkl')

def predict(input_data):
    input_data = np.array(input_data).reshape(-1, 1)  # Reshape to 2D array
    return model.predict(input_data)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.form['data']  # Get data from the form
    try:
        input_data = float(data)  # Convert to float
        prediction = predict([input_data])  # Call the prediction function
        return f"Prediction: {prediction[0]}"
    except ValueError:
        return "Invalid input! Please enter a number."

if __name__ == '__main__':
    app.run(debug=True, port=5002)  # Run the app on port 5001
