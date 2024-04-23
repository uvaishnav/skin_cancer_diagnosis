import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skinCancerDiagnosis.entity.config_entity import DataPrepConfig

class DataGenerator:
    def __init__(self, config:DataPrepConfig):
        self.config = config
        self.datagen = ImageDataGenerator(
            rescale = 1/255.0
        )
        self.train_dir = os.path.join(config.data_path,"train")
        self.test_dir = os.path.join(config.data_path,"test")
        self.val_dir = os.path.join(config.data_path,"val")

    def make_generator(self,datagen,file_path,shuffle):
        generator = datagen.flow_from_directory(
            file_path,
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            class_mode = self.config.params_class_mode,
            shuffle = shuffle
        )

        return generator
    
    def get_train_generator(self):
        if self.config.params_augmentation=="not":
            train_generator = self.make_generator(datagen=self.datagen,file_path=self.train_dir,shuffle=True)
        elif self.config.params_augmentation=="rand_aug":
            train_datagen = ImageDataGenerator(
                rescale = 1/255.0,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            train_generator = self.make_generator(datagen=train_datagen,file_path=self.train_dir,shuffle=True)
        else:
            train_generator = None

        return train_generator
    
    def get_test_generator(self):
        test_generator = self.make_generator(datagen=self.datagen,file_path=self.test_dir,shuffle=False)
        return test_generator
    
    def get_val_generator(self):
        val_generator = self.make_generator(datagen=self.datagen,file_path=self.val_dir,shuffle=False)
        return val_generator