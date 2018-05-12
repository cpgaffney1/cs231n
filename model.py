from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, concatenate
from keras.models import Model
from keras.applications.xception import Xception

class Config:
    numeric_input_size = 2
    img_shape = (299,299,3)
    n_classes = 1000
    batch_size = 32
    def __init__(self):
        pass

def build_model(config):
    numeric_inputs = Input(shape=(config.numeric_input_size,))
    img_inputs = Input(shape=config.img_shape)

    #running cnn
    cnn_out = Xception(include_top=False, weights='imagenet', input_tensor=img_inputs, input_shape=config.img_shape,
                                         pooling=None, classes=config.n_classes)(img_inputs)
    cnn_out = Flatten()(cnn_out)
    x = Dense(128, activation='relu')(cnn_out)
    x = Dense(64, activation='relu')(x)
    cnn_out = Dense(64, activation='relu')(x)

    #running fc
    x = Dense(64, activation='relu')(numeric_inputs)
    x = Dense(64, activation='relu')(x)
    fc_out = Dense(64, activation='relu')(x)

    #running RNN
    #MISSING
    #rnn_out
    # to top layer of text network

    #concat them
    x = concatenate([cnn_out, fc_out])#, rnn_out])

    #Final Fc
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    #Output Layer #Check if linear is a valid activation
    predictions = Dense(config.n_classes, activation='linear', name='main_output')(x)

    #Define Model 3 inputs and 1 output (Missing Rnn Input)
    model = Model(inputs=[numeric_inputs, img_inputs], outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model
