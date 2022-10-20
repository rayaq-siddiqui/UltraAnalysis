from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S
from tensorflow.keras.models import Model
from ClassWeightMult import ClassWeightMult


def efficient_net_v2s(class_weight, freeze_layers=400, input_shape=(448,448,3)):
    model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=input_shape)
    
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(7, activation='softmax')(x)
    out = ClassWeightMult(class_weight)(out)

    model_final = Model(model.input, out)
    for layer in model.layers[0:freeze_layers]:
        layer.trainable = False

    print(model_final.summary())
    return model_final
