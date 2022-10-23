import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image


class SegmentationDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col, batch_size=32, shuffle=True):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
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

            out.append(im)

        return out


    def __get_output(self, labels):
        out = []

        for label in labels:
            im = Image.open(label)
            im = im.crop((57,0,505,448))
            im = np.array(im)/1.
            im = im.reshape((448,448,1))
            # im = np.broadcast_to(im,(448,448,3))
            
            # print('out', im.shape)
            # im = im[:,:,0]
            
            out.append(im)

        return out


    def __get_data(self, batches):
        paths = batches[self.X_col]
        X = self.__get_input(paths)
        labels = batches[self.y_col]
        y = self.__get_output(labels)
        X = np.array(X, dtype=object)
        y = np.array(y, dtype=object)
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        return X,y


    def __getitem__(self, index):
        batches = self.df.iloc[index*self.batch_size:(index+1)*self.batch_size]
        X,y = self.__get_data(batches)
        return X,y


    def __len__(self):
        return self.n // self.batch_size
