import pandas as pd
import uuid
import datetime
from .create_logger import create_logger
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
        latest_dataset_name = self.registry.tail(1)['file_path'].values[0]

        # Read the latest dataset
        latest_dataset = pd.read_csv(path.join(datasets_path, f'{latest_dataset_name}'))

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
            id = str(uuid.uuid4())

            # Create a timestamp to record when the data was added
            now = datetime.datetime.now().isoformat()

            # Create a name for the dataset
            dataset_name = f'credit_{id}.csv'

            # Save the dataset
            merged_dataset.to_csv(path.join(datasets_path, dataset_name), index=False)

            # Add the merged dataset to the registry
            self.registry = pd.concat([self.registry, pd.DataFrame({ 
                'file_path': [dataset_name],
                'date_uploaded': [now]
            })])

            # Commit the registry
            self.registry.to_csv(path.join(datasets_path, 'registry.csv'), index=False)

            # Log
            self.__logger.info(f"Data merged succesfully into new dataset: {dataset_name}.")

            # Return true if all went well
            return True
        except Exception as e:
            print(e)
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
