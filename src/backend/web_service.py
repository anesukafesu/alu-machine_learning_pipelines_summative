from flask import Flask, render_template, request, jsonify, Response
from .model_manager import ModelsManager
from .data_manager import DataManager
from ...config import template_directory_path, static_assets_directory_path
from io import StringIO
import pandas as pd

app = Flask(
    __name__,
    template_folder=template_directory_path,
    static_folder=static_assets_directory_path
)

model_manager = ModelsManager()
data_manager = DataManager()

@app.route('/')
def index():
    return render_template("index.html")


@app.route("/data", methods=['POST'])
def data():
    # Get request body from request object
    request_data = request.get_data(as_text=True)

    # Convert the data into a stringio object
    request_data_io = StringIO(request_data)

    # Read the data into .csv
    new_dataset = pd.read_csv(request_data_io, header=False)

    if data_manager.add_data(new_dataset):
        # If add_data returns succesfully, we return with a 200 status code
        return Response(status=200)
    else:
        # Else we return with a 400 status code
        return Response(status=400)

@app.route('/predict')
def predict():
    # Extract the features
    person_home_ownership = request.args.get('person_home_ownership')
    loan_int_rate = int(request.args.get('loan_int_rate')) / 100
    loan_percent_income = int(request.args.get('loan_percent_income')) / 100
    cb_person_default_on_file = request.args.get('cb_person_default_on_file')

    # Make prediction
    prediction = model_manager.predict(
        person_home_ownership,
        loan_int_rate,
        loan_percent_income,
        cb_person_default_on_file
    )

    # Return the result
    return jsonify({ "prediction": 1 if prediction > 0.5 else 0 })

@app.route("/train")
def train():
    return jsonify(model_manager.train(data_manager))

@app.route('/evaluate')
def evaluate():
    return jsonify({
        'accuracy': model_manager.best_model_accuracy,
        'f1': model_manager.best_model_f1,
        'precision': model_manager.best_model_precision,
        'recall': model_manager.best_model_recall
    })


if __name__ == "__main__":
    app.run(debug=True)