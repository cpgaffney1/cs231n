from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, concatenate
from keras.models import Model

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
    i1 = Conv2D(32, (5, 5), padding='same', activation='relu')(img_inputs)
    i2 = Conv2D(32, (3, 3), padding='same', activation='relu')(i1)
    i3 = MaxPool2D((3, 3), strides=(1, 1), padding='same')(i2)
    i_final = Flatten()(i3)
    print(i_final)
    print(n_final)

    # to top layer of text network

    #concat them
    c1 = concatenate([n_final, i_final])
    c2 = Dense(64, activation='relu')(c1)
    predictions = Dense(1)(c1)

    model = Model(inputs=(numeric_inputs, img_inputs), outputs=predictions)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model