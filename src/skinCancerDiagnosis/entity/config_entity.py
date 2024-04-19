from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    dataset_name : str
    local_data_file : Path
    unzip_dir : Path

@dataclass(frozen=True)
class DataPrepConfig:
    data_path : Path
    params_image_size : list
    params_batch_size : int
    params_class_mode : str
    params_augmentation :str