import json
from typing import Optional
import numpy as np
import pandas as pd
import zipfile
import sys
import shutil
from src.Emotion_Detection.config.mongo_db_connection import MongoDBClient
from src.Emotion_Detection.constants.database import DATABASE_NAME
from src.Emotion_Detection.exception import SensorException
import os


class Imagedata:
    """
    This class helps to export entire MongoDB record as a Pandas DataFrame
    """

    def __init__(self):
        """
        Initialize MongoDB connection
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise SensorException(e, sys)
        
    def download_zipped_image_files(self, collection_name: str, directory_path: str, database_name: Optional[str] = None):
        """
        Retrieve zipped image files from MongoDB and save them as a single zip file preserving folder structure
        """
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            # Find all documents containing zipped image data
            documents = collection.find()

            # Create a temporary directory to store the images
            temp_dir = os.path.join(directory_path, 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Iterate through each document and retrieve the zipped image data
            for document in documents:
                # Retrieve the zipped image data from the document
                zipped_image_data = document.get('image_data')
                filename = document.get('filename')

                # Extract the folder name from the filename
                folder_name = '/'.join(filename.split('/')[:2])

                # Construct the directory path for saving the zipped image file
                folder_path = os.path.join(temp_dir, folder_name)

                # Create the directory if it does not exist
                os.makedirs(folder_path, exist_ok=True)

                # Write the zipped image data to the specified file path
                file_path = os.path.join(folder_path, os.path.basename(filename))
                with open(file_path, 'wb') as file:
                    file.write(zipped_image_data)

            # Create the final zip file with preserved folder structure
            with zipfile.ZipFile(os.path.join(directory_path, 'data.zip'), 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))

            # Clean up temporary directory
            shutil.rmtree(temp_dir)

            return os.path.join(directory_path, 'data.zip')
        except Exception as e:
            raise SensorException(e, sys)

    def save_zipped_image_data(self, zip_file_path: str, collection_name: str, database_name: Optional[str] = None):
        """
        Save zipped image data to MongoDB
        """
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_files = zip_ref.namelist()
                for filename in zip_files:
                    with zip_ref.open(filename) as file:
                        # Read image data from the zip file
                        image_data = file.read()

                        # Convert image data to a dictionary
                        record = {'filename': filename, 'image_data': image_data}

                        # Insert the record into MongoDB
                        if database_name is None:
                            collection = self.mongo_client.database[collection_name]
                        else:
                            collection = self.mongo_client[database_name][collection_name]
                        collection.insert_one(record)
            
            return len(zip_files)
        except Exception as e:
            raise SensorException(e, sys)
