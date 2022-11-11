import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image


class CNNDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col, batch_size=16, im_size=(448,448,3), shuffle=True):
        self.df = df.copy()
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(self.df)
        self.im_size = im_size


    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    

    def __get_input(self, paths):
        out = []

        for path in paths:
            im = Image.open(path)
            im = im.crop((57,0,505,448))
            im = np.array(im)/255.
            im = np.resize(im, self.im_size)

            out.append(im)

        return out


    def __get_output(self, labels):
        out = []

        for label in labels:
            out.append(label)

        return out


    def __get_data(self, batches):
        paths = batches[self.X_col]
        X = self.__get_input(paths)
        labels = batches[self.y_col]
        y = self.__get_output(labels)

        X = np.array(X)
        y = np.array(y)
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.int32)
        return X,y


    def __getitem__(self, index):
        batches = self.df.iloc[index*self.batch_size:(index+1)*self.batch_size]
        X,y = self.__get_data(batches)
        return X,y


    def get_all_data(self):
        paths = self.df[self.X_col]
        labels = self.df[self.y_col]

        outX = []

        for path in paths:
            im = Image.open(path)
            im = im.crop((57,0,505,448))
            im = np.array(im)/255.
            im = np.resize(im, self.im_size)

            outX.append(im)

        outy = []

        for label in labels:
            outy.append(label)

        return np.array(outX), outy


    def get_all_X(self):
        self.on_epoch_end()
        paths = self.df[self.X_col]
        outX = []

        for path in paths:
            im = Image.open(path)
            im = im.crop((57,0,505,448))
            im = np.array(im)/255.
            im = np.resize(im, self.im_size)

            outX.append(im)

        return np.array(outX)


    def get_all_y(self):
        labels = self.df[self.y_col]
        outy = []

        for label in labels:
            outy.append(label)

        return outy


    def __len__(self):
        return self.n // self.batch_size
