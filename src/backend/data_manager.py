import pandas as pd
import uuid
import datetime
from .create_logger import create_logger
from .visualiser import Visualiser
from os import path
from ...config import static_assets_directory_path, datasets_path
from .encoders import person_home_ownership_map, cb_person_default_on_file_map

class DataManager:
    """ Manages the applications data
    """
    def __init__(self):
        self.registry = pd.read_csv(path.join(datasets_path, 'registry.csv'))
        self.__logger = create_logger('data_manager')


    def get_latest_dataset(self):
        """ Returns the latest dataset
        """
        # Get the name of the latest dataset
        latest_dataset_name = self.registry.tail(1)['file_path'][0]

        # Read the latest dataset
        latest_dataset = pd.read_csv(path.join(datasets_path, f'{latest_dataset_name}.csv'))

        # Return the latest dataset
        return latest_dataset

    def add_data(self, new_dataset: pd.DataFrame):
        """ Creates a new dataset by merging existing data and the new data
        """
        try:
            # The data will only be added if it is valid
            new_dataset = self.__preprocess_data(new_dataset)

            # Read the latest dataset
            latest_dataset = self.get_latest_dataset()

            # Merge with the new dataset
            merged_dataset = pd.concat([latest_dataset, new_dataset]) 

            # Generate a unique id to be used in identifying the merged data
            id = str(uuid.uuid())

            # Create a timestamp to record when the data was added
            now = datetime.datetime.now().isoformat()

            # Create a name for the dataset
            dataset_name = f'credit_{id}.csv'

            # Save the dataset
            merged_dataset.to_csv(path.join(datasets_path, 'f{dataset_name}'), index=False)

            # Add the merged dataset to the registry
            self.registry = pd.concat(self.registry, [dataset_name, now])

            # Commit the registry
            self.registry.to_csv(path.join(datasets_path, 'registry.csv'), index=False)

            # Log
            self.__logger.info(f"Data merged succesfully into new dataset: {dataset_name}.")

            # Generate new visualisations
            # Define the visualisations
            visualisations = {
                'correlation_heatmap': Visualiser.create_class_distribution_bar_graph,
                'loan_interest_histogram': Visualiser.create_loan_interest_histogram,
                'home_ownership_piechart': Visualiser.create_piechart_showing_home_ownership,
                'loan_status_bargraph': Visualiser.create_class_distribution_bar_graph,
            }

            # For each visualisation, call the corresponding function
            for key, value in visualisations.items():
                image_file_path = path.join(static_assets_directory_path, 'images', f'{key}.png')
                value(merged_dataset, image_file_path)

            # Return true if all went well
            return True
        except Exception as e:
            self.__logger.error(e)
            return False
    
    def __preprocess_data(self, data: pd.DataFrame):
        """ Cleans the data by doing the following
        1. Drops unnecessary columns
        2. Ensure the required columns are there
        3. Encode values where necessary 
        """
        data = data[[
            'person_home_ownership',
            'loan_int_rate',
            'loan_status',
            'loan_percent_income',
            'cb_person_default_on_file'
        ]]

        data = data.replace({ 
            'person_home_ownership': person_home_ownership_map,
            'cb_person_default_on_file': cb_person_default_on_file_map
        })

        data = data.dropna()

        return data
