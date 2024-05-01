import os
import zipfile
from src.Emotion_Detection import logger
from src.Emotion_Detection.constants.database import COLLECTION_NAME
from src.Emotion_Detection.data_access.ImageData import Imagedata
from src.Emotion_Detection.entity.config_entity import DataIngestionConfig
from pathlib import Path
from src.Emotion_Detection.utils.common import get_size

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
     
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            push_URL = Path(self.config.push_URL)
            zip_download_dir = self.config.root_dir
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from Mongo DB into file {zip_download_dir}")

            image_data=Imagedata()
            zip_path=image_data.download_zipped_image_files(collection_name=COLLECTION_NAME,directory_path=zip_download_dir)
            print(zip_path)


            logger.info(f"Downloaded data from Mongo-DB into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)