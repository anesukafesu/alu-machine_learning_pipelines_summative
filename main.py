#!/usr/bin/env python3
from flask import Flask, render_template, request
from src.backend.model_manager import ModelsManager
from src.backend.data_manager import DataManager
from src.backend.config import template_directory_path, static_assets_directory_path
from src.backend.encoders import person_home_ownership_map
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import path

app = Flask(
    __name__,
    template_folder=template_directory_path,
    static_folder=static_assets_directory_path
)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

model_manager = ModelsManager()
data_manager = DataManager()

@app.route('/')
def index():
    return render_template("index.html")


@app.route("/data", methods=['POST'])
def data():
    if request.files.get('csv-data-file'):
        # Read the file from the frontend
        file = request.files.get('csv-data-file')

        # Convert the file into pandas dataframe
        new_dataset = pd.read_csv(file)

        if data_manager.add_data(new_dataset):
            # If add_data returns succesfully, we return with a 200 status code
            return render_template('data-success.html')
        else:
            # Else we return with a 400 status code
            return render_template('data-fail.html')
    
    if request.form.get('csv-data'):
        # Get request body from request object
        request_data = request.form.get('csv-data')

        # Convert the data into a stringio object
        request_data_io = StringIO(request_data)

        # Read the data into .csv
        new_dataset = pd.read_csv(request_data_io)

        if data_manager.add_data(new_dataset):
            # Create new visualisations
            create_visualisations()

            # If add_data returns succesfully, we return with a 200 status code
            return render_template('data-success.html')
        else:
            # Else we return with a 400 status code
            return render_template('data-fail.html')
    
    return render_template('data-fail.html')

def create_visualisations():
    # Get the latest data from the data_manager
    data = data_manager.get_latest_dataset()

    # Create a correlation heatmap
    corr = data.corr()
    save_file_path = path.join(static_assets_directory_path, 'images', 'correlation_heatmap.png')
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, vmin=-1, vmax=1, fmt=".2f", cmap='coolwarm', annot=True)
    plt.title('Correlation Heatmap')
    plt.savefig(save_file_path)

    # Create a home_ownership piechart
    save_file_path = path.join(static_assets_directory_path, 'images', 'home_ownership_piechart.png')
    counts = data['person_home_ownership'].value_counts()
    plt.figure(figsize=(10, 8))
    person_home_ownership_map_inverted = {v: k for k, v in person_home_ownership_map.items()}
    plt.pie(counts, labels=[person_home_ownership_map_inverted[index] for index in counts.axes[0]])
    plt.savefig(save_file_path)

    # Create a loan_interest histogram
    save_file_path = path.join(static_assets_directory_path, 'images', 'loan_interest_histogram.png')
    plt.figure(figsize=(10, 8))
    plt.hist(data['loan_int_rate'])
    plt.savefig(save_file_path)

    # Create a loan_status barplot
    save_file_path = path.join(static_assets_directory_path, 'images', 'loan_status_bargraph.png')
    labelled_data = data['loan_status']
    value_counts = labelled_data.value_counts()
    default_count = value_counts.get(1, 0)
    not_in_default_count = value_counts.get(0, 0)
    categories = ['In default', 'Not in default']
    counts = [default_count, not_in_default_count]
    plt.figure(figsize=(10, 8))
    plt.bar(categories, counts)
    plt.savefig(save_file_path)

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
    return render_template('prediction.html', prediction=1 if prediction > 0.5 else 0)

@app.route("/train")
def train():
    metrics = model_manager.train(data_manager)
    return render_template('model_metrics.html', metrics=metrics)

@app.route('/evaluate')
def evaluate():
    metrics = {
        'accuracy': model_manager.best_model_accuracy,
        'f1': model_manager.best_model_f1,
        'precision': model_manager.best_model_precision,
        'recall': model_manager.best_model_recall
    }
    return render_template('model_metrics.html', metrics=metrics)


if __name__ == "__main__":
    app.run()