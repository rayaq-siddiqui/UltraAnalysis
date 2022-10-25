from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, BatchNormalization, Input, Conv2D, Flatten
from tensorflow.keras.models import Model
from .ClassWeightMult import ClassWeightMult


def simple_seq_model(class_weight, input_shape=(448,448,3)):
    # sequential attempt
    inp =  Input(input_shape)
    x = BatchNormalization()(inp)
    
    x = Conv2D(64, kernel_size=6, strides=(3,3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(64, kernel_size=6, strides=(3,3), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(64, kernel_size=3, strides=(1,1), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(64, kernel_size=3, strides=(1,1), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(3, activation='relu')(x)
    out = ClassWeightMult(class_weight)(x)
    out = Dense(3, activation='softmax')(out)
    
    model = Model(inp, out)

    print(model.summary())
    return model
