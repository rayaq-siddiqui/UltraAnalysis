from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, BatchNormalization, Input, Conv2D, Flatten
from tensorflow.keras.models import Model
from ClassWeightMult import ClassWeightMult


def seq_model(class_weight):
    # sequential attempt
    inp =  Input((448,448,3))
    x = BatchNormalization()(inp)
    
    x = Conv2D(64, kernel_size=5, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    x = Conv2D(64, kernel_size=5, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)

    x = Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    x = Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    x = Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    
    x = Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    x = Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(7, activation='softmax')(x)
    out = ClassWeightMult(class_weight)(x)
    
    model = Model(inp, out)

    print(model.summary())
    return model
