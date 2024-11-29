from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from .create_logger import create_logger
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
import pickle
import uuid
import datetime
from .encoders import person_home_ownership_map, cb_person_default_on_file_map
from ...config import models_path
from os import path, makedirs


class ModelsManager:
    def __init__(self):
        self.__logger = create_logger("models_manager")
        self.__registry_path = path.join(models_path, "registry.csv")
        self.__registry = pd.read_csv(self.__registry_path)
        self.__best_model_id = self.__get_best_model_id()

        # Get the best_model_id and performance metrics
        best_model_record = self.__registry.iloc[self.__best_model_id]
        best_model_id = best_model_record['id']
        self.best_model_accuracy = best_model_record['accuracy']
        self.best_model_f1 = best_model_record['f1']
        self.best_model_precision = best_model_record['precision']
        self.best_model_recall = best_model_record['recall']

        best_model_path = self.__create_model_path(best_model_id)
        best_model_scaler_path = self.__create_scaler_path(best_model_id)

        with open(best_model_path, 'rb') as f:
            self.__model = pickle.load(f)

        with open(best_model_scaler_path, 'rb') as f:
            self.__scaler = pickle.load(f)

    def train(self, data_manager):
        """ Trains a machine learning model using data stored in data
        """
        # Get the latest dataset for training
        df: pd.DataFrame = data_manager.get_latest_dataset()

        # Split the data into features and target
        y = df['loan_status']
        X = df.drop('loan_status', axis=1)

        # Scale the features for better training performance
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Balance the dataset
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, y)

        # Split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LinearRegression()

        # Training the model
        model.fit(X_train, y_train)

        # Evaluating the model
        y_pred = (model.predict(X_test) > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Save the model with metrics
        self.__save_model(model, scaler, accuracy, f1, precision, recall)

        # Return the statistics
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def predict(self, person_home_ownership, loan_int_rate, loan_percent_income, cb_person_default_on_file):
        """ Make predictions on the data
        """
        # Encode the features
        person_home_ownership_encoded = person_home_ownership_map[person_home_ownership]
        cb_person_default_on_file_encoded = cb_person_default_on_file_map[cb_person_default_on_file]

        # Put the features together into an array
        features = pd.DataFrame({
            'person_home_ownership': [person_home_ownership_encoded],
            'loan_int_rate': [loan_int_rate],
            'loan_percent_income': [loan_percent_income],
            'cb_person_default_on_file': [cb_person_default_on_file_encoded]
        })

        # Scale the features using the scaler for the model we are using to predict
        features = self.__scaler.transform(features)

        # Make a prediction
        return float(self.__model.predict(features)[0])

    def __save_model(self, model, scaler, accuracy, f1, precision, recall):
        """ Saves a model in the models folder and registry
        """
        try:
            # Log beginning
            self.__logger.info("Saving new model.")

            # Generate a unique id to be used to identify the model
            id = str(uuid.uuid4())

            # Creating the model and scaler paths path.
            model_path = self.__create_model_path(id)
            scaler_path = self.__create_scaler_path(id)

            # Create directories to save models in
            makedirs(path.dirname(model_path), exist_ok=True)

            # Save the model using pickle.dump.
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            
            # Save the scaler using pickle.dump.
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            # Generate a datetime for the creation of the model.
            date = datetime.datetime.now().isoformat()
            
            # Add the model to the registry, and save it.
            self.__add_model_to_registry(id, date, accuracy, f1, precision, recall)

            # Compare the model to current best using f1 score
            # Get best f1_score
            best_f1 = self.__registry.iloc[self.__best_model_id]['f1']

            # If the new score exceeds the best, save the new as the best
            # The latest model is the last one added to the registry
            # It's row index will be equal to the number of rows in the dataset
            # Probably not the best way to do this, it feels hacky and vulnerable
            # But it will have to do for now.
            if f1 > best_f1:
                self.__model = model
                self.__scaler = scaler
                self.__best_model_id = len(self.__registry.index)
                self.__save_best_model_id()

        except Exception as e:
            self.__logger.error("Something went wrong saving the model. " + e)
    
    def __get_best_model_id(self):
        """ Retrieves the best model id from file and returns it.
        """
        best_model_txt_file_path = path.join(models_path, 'best_model.txt')
        with open(best_model_txt_file_path, 'r') as f:
            return int(f.readline())

    def __save_best_model_id(self):
        """ Writes whichever model id is saved as the self.__best_model_id to file
        """
        best_model_txt_file_path = path.join(models_path, 'best_model.txt')
        with open(best_model_txt_file_path, 'w') as f:
            f.write(str(self.__best_model_id))
    
    def __add_model_to_registry(self, id, date_trained, accuracy, f1, precision, recall):
        """ Add the model to the registry
        """
        # Add model to pandas dataframe
        self.__registry = pd.concat([
            self.__registry,
            pd.DataFrame({
                'id': [id],
                'date_trained': [date_trained],
                'accuracy': [accuracy],
                'f1': [f1],
                'precision': [precision],
                'recall': [recall]
            })
        ])

        # Save the dataframe as csv
        self.__registry.to_csv(self.__registry_path, index=False)
    
    def __create_model_path(self, id):
        return path.join(models_path, id, 'model.pkl')

    def __create_scaler_path(self, id):
        return path.join(models_path, id, 'scaler.pkl')