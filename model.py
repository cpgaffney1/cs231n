from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, concatenate, LSTM, BatchNormalization
from keras.models import Model
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.layers import Embedding

class Config:
    numeric_input_size = 2
    img_shape = (224,224,3)
    n_classes = 1000
    batch_size = 64
    embed_dim = 50
    max_seq_len = 30
    def __init__(self, word_index, embedding_matrix, lr=0.001, n_recurrent_layers=2, n_numeric_layers=1,
                 trainable_convnet_layers=20, imagenet_weights=True, n_top_hidden_layers=1, n_convnet_fc_layers=3,
                 n_classes=1000, drop_prob=0.0):
        self.word_index = word_index
        self.embedding_matrix = embedding_matrix
        self.vocab_size = len(word_index)
        self.embed_dim = embedding_matrix.shape[1]
        self.text_shape = (self.max_seq_len, self.vocab_size)
        self.lr = lr
        self.n_recurrent_layers = n_recurrent_layers
        self.n_numeric_layers = n_numeric_layers
        self.trainable_convnet_layers = trainable_convnet_layers
        self.imagenet_weights = imagenet_weights
        self.n_top_hidden_layers = n_top_hidden_layers
        self.n_convnet_fc_layers = n_convnet_fc_layers
        self.n_classes = n_classes
        self.drop_prob = drop_prob

def build_model(config):
    numeric_inputs = Input(shape=(config.numeric_input_size,))
    img_inputs = Input(shape=config.img_shape)
    #text_inputs = Input(shape=config.text_shape)

    #running cnn
    #image_model = Xception(include_top=False, weights='imagenet', input_tensor=img_inputs, input_shape=config.img_shape,
    #                                     pooling=None, classes=config.n_classes)
    if config.imagenet_weights:
        weights = 'imagenet'
    else:
        weights = None
    image_model = Xception(input_shape=config.img_shape, include_top=False, weights=weights,
                        input_tensor=img_inputs, classes=config.n_classes)
    #freeze lower layers
    if weights is not None:
        for i in range(len(image_model.layers) - config.trainable_convnet_layers):
            image_model.layers[i].trainable = False

    cnn_out = image_model(img_inputs)
    cnn_out = Flatten()(cnn_out)
    x = Dense(128, activation='relu')(cnn_out)
    for i in range(config.n_convnet_fc_layers):
        x = Dense(64, activation='relu')(x)
    cnn_out = x

    #running fc
    x = Dense(64, activation='relu')(numeric_inputs)
    for i in range(config.n_numeric_layers - 1):
        x = Dense(64, activation='relu')(x)
    fc_out = x
    #running RNN
    '''embedding_layer = Embedding(len(config.word_index) + 1, config.embed_dim,
                                weights=[config.embedding_matrix],
                                input_length=config.max_seq_len,
                                trainable=False)
    embedded_seqs = embedding_layer(text_inputs)
    lstm = LSTM(128)(embedded_seqs)
    for i in range(config.n_recurrent_layers - 1):
        lstm = LSTM(64)(lstm)
    rnn_out = lstm'''
    #rnn_out
    # to top layer of text network

    #concat them
    x = concatenate([cnn_out, fc_out])#, rnn_out])

    #Final Fc
    #for i in range(config.n_top_hidden_layers):
    #    x = Dense(64, activation='relu')(x)

    predictions = Dense(config.n_classes, activation='softmax', name='main_output')(x)

    #Define Model 3 inputs and 1 output (Missing Rnn Input)
    model = Model(inputs=[numeric_inputs, img_inputs], outputs=predictions)
    opt = Adam(lr=config.lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy'])
    return model

import os
import pickle
def write_model(model, config, best_val_loss, model_folder):
    model.save(model_folder + 'model.h5'.format(best_val_loss))
    with open(model_folder + 'config'.format(best_val_loss), 'wb') as pickle_file:
        pickle.dump(config, pickle_file)


from keras.models import load_model as load_keras_model
def load_model(model_folder):
    path = 'models/' + model_folder + '/'
    model = load_keras_model(path + 'model.h5')
    with open(path + 'config', 'rb') as pickle_file:
        config = pickle.load(pickle_file)
    return config, model

