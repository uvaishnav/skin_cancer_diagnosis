import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


from skinCancerDiagnosis import logger
from skinCancerDiagnosis.utils.common import get_size
from skinCancerDiagnosis.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config : DataIngestionConfig):
        self.config = config

    def download_file(self)->str:

        # fetch data from kaggle
        try:
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()

            # Specify the dataset you want to download
            dataset_name = self.config.dataset_name

            # Download dataset
            zip_download_dir = self.config.local_data_file
            api.dataset_download_files(dataset_name, path=zip_download_dir)

            logger.info(f"Downloaded data from {dataset_name} into file {zip_download_dir}")

        except Exception as e:
            logger.exception(e)
            raise e
    

    def extract_zip_file(self):

        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        zip_data = os.listdir(self.config.local_data_file)
        logger.info("we have {} in {}".format(zip_data,self.config.local_data_file))
        zip_data_path = os.path.join(self.config.local_data_file, zip_data[0])

        with zipfile.ZipFile(zip_data_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        

    




