import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
import os
from sklearn.model_selection import train_test_split


# this project imports
from data_generator.sklearn_data_generator import SklearnDataGenerator
from models import (
    svm,
    rf,
    knn
)


def train_sklearn(_model, _verbose):
    # mapping for the labels for the cnn
    class_mapping = {
        'benign': 0,
        'malignant': 1,
        'normal': 2,
    }

    # getting all the class folders
    data_path_img = 'cnn_data/'
    folders = os.listdir(data_path_img)

    # creating the path, class dataset
    paths, classes = [], []
    for folder in folders:
        for file in os.listdir(data_path_img+folder):
            path = f"{data_path_img}{folder}/{file}"
            _class = class_mapping[folder]
            paths.append(path)
            classes.append(_class)

    # splitting the data into testing and training
    X_train, X_test, y_train, y_test = train_test_split(
        paths, classes,
        test_size=.33,
        random_state=42
    )

    # creating a dataframe for the generator
    train_df = pd.DataFrame({'path': X_train, 'class': y_train})
    val_df = pd.DataFrame({'path': X_test, 'class': y_test})

    # creating the train/val generator
    traingen = SklearnDataGenerator(train_df, 'path', 'class', BATCH_SIZE)
    valgen = SklearnDataGenerator(val_df, 'path', 'class', BATCH_SIZE)

    # get the model
    model = None

    if _model == 'rf':
        model = rf.rf()
    elif _model == 'knn':
        model = svm.svm()
    elif _model == 'svm':
        model = knn.knn()

    # fit the model
    train_X, train_y = traingen.get_all_data()
    svm.fit(train_X, train_y)

    return model, traingen, valgen
