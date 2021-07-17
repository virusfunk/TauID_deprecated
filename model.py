from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, MaxPool2D, Dropout, Conv2D, Activation, Input, Flatten, Add

def get_baseline():
    input_images = Input(shape=(256,256,4))

    x = Conv2D(32, 3, 1)(input_images)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Conv2D(64, 3, 1)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Conv2D(128, 3, 1)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)

    x = Dense(64)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)

    x = Dense(32)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    
    output_onehots = Dense(6, activation='softmax')(x)
    model = Model(inputs=input_images, outputs=output_onehots)
    return model



