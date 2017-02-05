import os
import csv
import tqdm
import scipy.misc
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


class FER2013(object):
    """0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral"""
    def __init__(self, data_directory, target_size):
        self.n_class = 7
        self.target_size = target_size
        self.data_directory = data_directory
        self.train_directory = os.path.join(self.data_directory, "Training")
        self.test_directory =  os.path.join(self.data_directory, "PublicTest")
        self.valid_directory =  os.path.join(self.data_directory, "PrivateTest")

    def split(self):
        if not os.path.exists(self.train_directory):
            os.makedirs(self.train_directory)
            for i in range(self.n_class):
                os.makedirs(os.path.join(self.train_directory, str(i)))

        if not os.path.exists(self.test_directory):
            os.makedirs(self.test_directory)
            for i in range(self.n_class):
                os.makedirs(os.path.join(self.test_directory, str(i)))
        
        if not os.path.exists(self.valid_directory):
            os.makedirs(self.valid_directory)
            for i in range(self.n_class):
                os.makedirs(os.path.join(self.valid_directory, str(i)))
        
        
        data_path = os.path.join(self.data_directory, "fer2013.csv")
        with open(data_path, "r") as f:
            reader = csv.reader(f)

            # skip header
            next(reader, None)
                
            for i, (emotion, pixels, usage) in tqdm.tqdm(enumerate(reader), desc="splitting", total=35887):
                pixels = np.fromstring(pixels, sep=" ").reshape((48, 48))
                path = os.path.join(self.data_directory, usage, emotion, "{0}.jpg".format(i))
                scipy.misc.imsave(path, pixels)
        
    def train_generator(self, batch_size, **kwargs):
        datagen = ImageDataGenerator(**kwargs)
        return datagen.flow_from_directory(
            directory=self.train_directory, 
            target_size=self.target_size,
            color_mode="grayscale",
            batch_size=batch_size, 
            class_mode="categorical")
        
    
    def test_generator(self, batch_size, **kwargs):
        datagen = ImageDataGenerator(**kwargs)
        return datagen.flow_from_directory(
            directory=self.test_directory,
            target_size=self.target_size,
            color_mode="grayscale",
            batch_size=batch_size, 
            class_mode="categorical")
    
    def valid_generator(self, batch_size, **kwargs):
        datagen = ImageDataGenerator(**kwargs)
        return datagen.flow_from_directory(
            directory=self.valid_directory,
            target_size=self.target_size,
            color_mode="grayscale",
            batch_size=batch_size, 
            class_mode="categorical")
    