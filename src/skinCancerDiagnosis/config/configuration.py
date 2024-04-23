import os
from skinCancerDiagnosis.constants import *
from skinCancerDiagnosis.utils.common import read_yaml, create_directories
from skinCancerDiagnosis.entity.config_entity import (
    DataIngestionConfig,
    DataPrepConfig,
    TrailTrainingConfig,
    ModelevalConfig
)

class ConfugarationManager:
    def __init__(
            self,
            config_file_path = CONFIG_FILE_PATH,
            params_file_path = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            dataset_name = config.dataset_name,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_prep_config(self)->DataPrepConfig:

        config = self.config.data_prep

        data_prep_config = DataPrepConfig(
            data_path = config.data_path,
            params_image_size= self.params.IMAGE_SIZE,
            params_batch_size= self.params.BATCH_SIZE,
            params_class_mode= self.params.CLASS_MODE,
            params_augmentation= self.params.AUGMENTATION_TYPE
        )

        return data_prep_config
    
    def get_trail_training_config(self)->TrailTrainingConfig:

        config = self.config.trail_training
        create_directories([config.root_dir])

        trail_training_config = TrailTrainingConfig(
            root_dir = config.root_dir,
            params_epochs = self.params.EPOCHS,
            params_include_top = self.params.INCLUDE_TOP,
            params_weights = self.params.WEIGHTS,
            params_learning_rate = self.params.LEARNING_RATE,
            params_patience = self.params.PATIENCE
        )

        return trail_training_config
    
    def get_evaluation_config(self,model_path)->ModelevalConfig:

        config = self.config.trail_training

        path = os.path.join(config.root_dir,model_path)

        model_evaluation_config = ModelevalConfig(
            model_path= path,
            all_params= self.params,
            mlflow_uri= 'https://dagshub.com/uvaishnav/skin_cancer_diagnosis.mlflow'
        )

        return model_evaluation_config