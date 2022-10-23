import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image

class SklearnDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col, shuffle=True):
        self.df = df.copy()
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.X_col = X_col
        self.y_col = y_col
        self.shuffle = shuffle
        self.n = len(self.df)


    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    

    def __get_input(self, paths):
        out = []

        for path in paths:
            im = Image.open(path)
            im = im.crop((57,0,505,448))
            im = np.array(im)/255.
            im = np.reshape(im, im.shape[0]*im.shape[1]*im.shape[2])

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


    # def __getitem__(self, index):
    #     batches = self.df.iloc[index*self.batch_size:(index+1)*self.batch_size]
    #     X,y = self.__get_data(batches)
    #     return X,y


    def get_all_data(self):
        paths = self.df[self.X_col]
        labels = self.df[self.y_col]

        outX = []

        for path in paths:
            im = Image.open(path)
            im = im.crop((57,0,505,448))
            im = np.array(im)/255.
            im = np.reshape(im, im.shape[0]*im.shape[1]*im.shape[2])

            outX.append(im)

        outy = []

        for label in labels:
            outy.append(label)

        return np.array(outX), outy


    def __len__(self):
        return self.n
