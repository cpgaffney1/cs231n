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

    #running cnn
    cnn_out = Xception(include_top=False, weights='imagenet', input_tensor=img_inputs, input_shape=config.img_shape,
                                         pooling=None, classes=1000)(img_inputs)
    #running fc
    x = Dense(64, activation='relu')(numeric_inputs)
    x = Dense(64, activation='relu')(x)
    fc_out = Dense(64, activation='relu')(x)

    #running RNN
    #MISSING
    #rnn_out
    # to top layer of text network

    #concat them
    x = keras.layers.concatente([cnn_])
    c1 = concatenate([n_final, i_final])
    c2 = Dense(64, activation='relu')(c1)
    predictions = Dense(1)(c2)

    model = Model(inputs=(numeric_inputs, img_inputs), outputs=predictions)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
