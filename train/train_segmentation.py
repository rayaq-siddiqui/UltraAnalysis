import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
import os
from sklearn.model_selection import train_test_split

# this project imports
from data_generator.segmentation_data_generator import SegmentationDataGenerator
from models import (
    unet,
    enet,
)
from utils.diceloss import DiceLoss

def train_segmentation(_model, _verbose, _epochs, BATCH_SIZE=16, im_size=(448,448,3), load_weights=False):
    # getting the image and mask file paths
    img_path = 'data/benign/img/'
    mask_path = 'data/benign/mask/'
    
    img_files = os.listdir(img_path)
    img_files = [img_path+i for i in img_files]
    mask_files = os.listdir(mask_path)
    mask_files = [mask_path+i for i in mask_files]

    # splitting the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        img_files, mask_files,
        test_size=.33,
        random_state=42
    )

    # creating a dataframe for the generator
    train_df = pd.DataFrame({'img':X_train, 'labels':y_train})
    val_df = pd.DataFrame({'img':X_test, 'labels':y_test})

    # creating the generator
    traingen = SegmentationDataGenerator(train_df, 'img', 'labels', BATCH_SIZE)
    valgen = SegmentationDataGenerator(val_df, 'img', 'labels', BATCH_SIZE)

    print('data loaded and generators created')

    # getting the model
    model = None

    if _model == 'unet':
        model = unet.UNet(im_size)

        # getting params ready to compile model
        if load_weights and os.path.exists('checkpoints/unet/'):
            print('weights loaded')
            model.load_weights('checkpoints/unet/')
    elif _model == 'enet':
        model = enet.ENet(
            n_classes=1, 
            input_height=im_size[0], 
            input_width=im_size[1]
        )  

        # getting params ready to compile model
        if load_weights and os.path.exists('checkpoints/enet/'):
            print('weights loaded')
            model.load_weights('checkpoints/enet/')

    print('model loaded')
    print(model.summary())

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        epsilon=1e-07,
    )
    loss = DiceLoss()

    # compile model
    model.compile(
        optimizer=opt,
        loss=loss,
        # metrics=['accuracy']
    )

    print('model compiled')

    # creating callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"checkpoints/{_model}",
        verbose=1,
        monitor='loss',
        save_best_only=True
    )
    lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=10,
        verbose=1,
        min_delta=0.0001,
        cooldown=3,
        min_lr=1e-8,
    )
    callbacks=[checkpoint, lr_reduction]

    model.fit(
        traingen,
        epochs=_epochs,
        verbose=_verbose,
        callbacks=callbacks
    )

    print('model fitted complete')

    return model, traingen, valgen

