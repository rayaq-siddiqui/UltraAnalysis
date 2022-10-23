import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
import os
from sklearn.model_selection import train_test_split

# this project imports
from data_generator.cnn_data_generator import CNNDataGenerator
from models import (
    efficient_net_b7,
    efficient_net_v2s,
    inception_v3,
    resnet50,
    seq_model,
    simple_seq_model,
    vgg16
)
from utils.calcClassWeight import calcClassWeight


def train_cnn(_model, _verbose, _epochs, BATCH_SIZE=16, im_size=(448,448,3)):
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
    traingen = CNNDataGenerator(train_df, 'path', 'class', BATCH_SIZE)
    valgen = CNNDataGenerator(val_df, 'path', 'class', BATCH_SIZE)

    print('data loaded and generators created')

    # calculate the class_weights
    data_dir='cnn_data/'
    n_classes = im_size[-1]
    class_weight = calcClassWeight(n_classes, data_dir)

    # getting the model
    model = None

    if _model == 'efficient_net_b7':
        model = efficient_net_b7.efficient_net_b7(
            class_weight, input_shape=im_size
        )
    elif _model == 'efficient_net_v2s':
        model = efficient_net_v2s.efficient_net_v2s(
            class_weight, input_shape=im_size
        )
    elif _model == 'inception_v3':
        model = inception_v3.inception_v3(
            class_weight, input_shape=im_size
        )
    elif _model == 'resnet50':
        model = resnet50.resnet50(
            class_weight, input_shape=im_size,
        )
    elif _model == 'seq':
        model = seq_model.seq_model(
            class_weight, input_shape=im_size
        )
    elif _model == 'simple_seq':
        model = simple_seq_model.simple_seq_model(
            class_weight, im_size
        )
    elif _model == 'vgg16':
        model = vgg16.vgg16(
            class_weight, input_shape=im_size,
        )

    print('model loaded')

    # parameters for model compilation
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [
        'accuracy',
        # tfa.metrics.F1Score(3, 'micro')
    ]

    # compiling model
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )

    print('model compiled')

    # defining callbacks
    lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', 
        factor=0.5, 
        patience=3, 
        verbose=_verbose,
        cooldown=3, 
        min_lr=1e-10
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"checkpoints/{_model}/", 
        save_best_only=True,
        monitor='loss',
        verbose=_verbose
    )
    callbacks = [lr_reduction, checkpoint]

    model.fit(
        traingen,
        validation_data=valgen,
        epochs=_epochs,
        verbose=_verbose,
        callbacks=callbacks,
    )

    print('model fitted complete')

    return model, traingen, valgen
