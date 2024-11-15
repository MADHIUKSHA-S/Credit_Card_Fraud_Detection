from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the models
with open('models/dt_model.pkl', 'rb') as dt_file:
    dt_model = pickle.load(dt_file)

# Function to predict using Decision Tree model
def predict_decision_tree(input_data):
    input_array = np.array(input_data).reshape(1, -1)  # Reshape the input to match the model's expected input
    return dt_model.predict(input_array)[0]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get the input data from the form (time, V1 to V27, amount)
            input_data = request.form["input_data"]
            input_data_list = [float(i) for i in input_data.split(",")]  # Convert to list of floats
            
            if len(input_data_list) != 29:  # Check for correct number of inputs (time + V1 to V27 + amount = 29)
                return "Error: Please provide exactly 29 input values (time, V1 to V27, amount)."

            # Make prediction
            dt_result = predict_decision_tree(input_data_list)
            
            return f"Decision Tree Prediction: {dt_result}"
        except ValueError:
            return "Error: Could not convert input values to float. Ensure the inputs are valid numbers."
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
