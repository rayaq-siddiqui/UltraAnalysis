from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from .ClassWeightMult import ClassWeightMult


def inception_v3(class_weight, freeze_layers=250, input_shape=(448,448,3)):
    model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(3, activation='relu')(x)
    out = ClassWeightMult(class_weight)(x)
    out = Dense(3, activation='softmax')(out)

    model_final = Model(model.input, out)

    print(model_final.summary())
    return model_final
