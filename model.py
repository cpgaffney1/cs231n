from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, concatenate
from keras.models import Model
from keras.applications.xception import Xception

numeric_input_size = 2
img_shape = (300, 400, 3)

class Config:
    numeric_h_size = 10

    def __init__(self):
        pass

def build_model(config):
    numeric_inputs = Input(shape=(numeric_input_size,))
    img_inputs = Input(shape=img_shape)

    # propagate to top layer of numeric network
    n_final = Dense(config.numeric_h_size, activation='relu')(numeric_inputs)

    # to top layer of image network
    i1_1 = Conv2D(32, (5, 5), padding='same', activation='relu')(img_inputs)
    i1_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(i1_1)
    i1_3 = MaxPool2D((2, 2), strides=(2, 2), padding='same')(i1_2)

    i2_1 = Conv2D(32, (5, 5), padding='same', activation='relu')(i1_3)
    i2_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(i2_1)
    i2_3 = MaxPool2D((2, 2), strides=(2, 2), padding='same')(i2_2)

    i3_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(i2_3)
    i3_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(i3_1)
    i3_3 = MaxPool2D((2, 2), strides=(2, 2), padding='same')(i3_2)

    i_final = Flatten()(i3_3)

    Xception(include_top=False, weights='imagenet', input_tensor=img_inputs, input_shape=config.img_shape,
                                         pooling=None, classes=1000)

    # to top layer of text network

    #concat them
    c1 = concatenate([n_final, i_final])
    c2 = Dense(64, activation='relu')(c1)
    predictions = Dense(1)(c2)

    model = Model(inputs=(numeric_inputs, img_inputs), outputs=predictions)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
