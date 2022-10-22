from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications.efficientnet import EfficientNetB7 
from tensorflow.keras.models import Model
from .ClassWeightMult import ClassWeightMult


def efficient_net_b7(class_weight, freeze_layers=750, input_shape=(448,448,3)):
    model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
    
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(3, activation='softmax')(x)
    out = ClassWeightMult(class_weight)(x)

    model_final = Model(model.input, out)
    for layer in model.layers[0:freeze_layers]:
        layer.trainable = False

    print(model_final.summary())
    return model_final
