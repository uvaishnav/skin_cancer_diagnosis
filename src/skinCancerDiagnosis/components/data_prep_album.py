import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class DataAlbumPrep:
    def __init__(self,train_path:Path,test_path:Path,num_classes):
        self.train_dir = train_path
        self.test_dir = test_path
        self.num_classes = num_classes
        self.labels = os.listdir(train_path)
        self.transform = A.Compose([
            A.Rotate(limit=40),
            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.HorizontalFlip(),
        ])

        self.le = LabelEncoder()

    def normalize_data(self,x_data,y_data):
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        y_data = self.le.fit_transform(y_data) # to numerical
        y_data = to_categorical(y_data,self.num_classes)     # to one-hot-encoding
        x_data = x_data/255 

        return x_data,y_data

    def get_data(self,aug_status,data_path:Path,img_size:tuple):
        x_data = []
        y_data = []

        for label in self.labels:
            path = os.path.join(data_path,label)
            folder_data = os.listdir(path)
            for image_path in folder_data:
                image = cv2.imread(os.path.join(path,image_path), cv2.IMREAD_COLOR)
                if(aug_status==True):
                    transformed = self.transform(image=image)
                    transformed_image = transformed['image']
                    image_resized = cv2.resize(transformed_image,img_size)
                else:
                    image_resized = cv2.resize(image, img_size)
                
                x_data.append(np.array(image_resized))
                y_data.append(label)
        
        return x_data, y_data
    
    def get_train_data(self,img_size:tuple):
        x_data = []
        y_data = []

        # get Normal Data
        x,y = self.get_data(aug_status=False,data_path = self.train_dir,img_size=img_size)
        x_data.extend(x)
        y_data.extend(y)

        # get augmented data
        x,y = self.get_data(aug_status=True,data_path = self.train_dir,img_size=img_size)
        x_data.extend(x)
        y_data.extend(y)

        x_train, y_train = self.normalize_data(x_data=x_data,y_data=y_data)
        return (x_train,y_train)
    
    # def get_test_data(self,img_size:tuple):
    #     x_test, y_test = self.get_data(aug_status=False,data_path=self.test_dir,img_size=img_size)
    #     x_test, y_test = self.normalize_data(x_data=x_test,y_data=y_test)

    #     return (x_test,y_test)