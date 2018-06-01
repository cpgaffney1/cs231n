from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, concatenate, LSTM, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam, SGD
from keras.layers import Embedding
import keras.regularizers as regularizers
import keras.losses as losses
import keras.backend as K


class Config:
    numeric_input_size = 2
    img_shape = (224,224,3)
    n_classes = 100
    batch_size = 32
    embed_dim = 50
    max_seq_len = 30
    def __init__(self, word_index, embedding_matrix, tokenizer, lr=0.001, n_recurrent_layers=1, n_numeric_layers=3,
                 trainable_convnet_layers=20, imagenet_weights=True, n_top_hidden_layers=1, n_convnet_fc_layers=2,
                 n_classes=100, drop_prob=0.5, reg_weight=0.01, img_only=False, numeric_input_size=2, freeze_cnn=True,
                 numeric_only=False, rnn_only=False, distance_weight=0.001):
        self.word_index = word_index
        self.embedding_matrix = embedding_matrix
        self.vocab_size = len(word_index)
        self.embed_dim = embedding_matrix.shape[1]
        self.lr = lr
        self.n_recurrent_layers = n_recurrent_layers
        self.n_numeric_layers = n_numeric_layers
        self.trainable_convnet_layers = trainable_convnet_layers
        self.imagenet_weights = imagenet_weights
        self.n_top_hidden_layers = n_top_hidden_layers
        self.n_convnet_fc_layers = n_convnet_fc_layers
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.reg_weight = reg_weight
        self.tokenizer = tokenizer
        self.img_only=img_only
        self.numeric_input_size = numeric_input_size
        self.freeze_cnn = freeze_cnn
        self.numeric_only = numeric_only
        self.rnn_only = rnn_only
        self.distance_weight = distance_weight

def build_model(config):
    img_inputs = Input(shape=config.img_shape, name='img_input')
    numeric_inputs = Input(shape=(config.numeric_input_size,))
    text_inputs = Input(shape=(config.max_seq_len,))

    #running cnn
    if config.imagenet_weights:
        weights = 'imagenet'
    else:
        weights = None
    image_model = ResNet50(include_top=False, weights=weights)
    #freeze lower layers

    cnn_out = image_model.output
    x = GlobalAveragePooling2D()(cnn_out)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(config.reg_weight))(x)
    x = Dropout(config.drop_prob)(x)
    cnn_out = x

    #running fc
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(config.reg_weight))(numeric_inputs)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(config.reg_weight))(x)
    x = Dropout(config.drop_prob)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(config.reg_weight))(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(config.reg_weight))(x)
    x = Dropout(config.drop_prob)(x)
    fc_out = x

    #running RNN
    embedding_layer = Embedding(len(config.word_index) + 1, config.embed_dim,
                                weights=[config.embedding_matrix],
                                trainable=False)
    embedded_seqs = embedding_layer(text_inputs)
    lstm = LSTM(64)(embedded_seqs)
    for i in range(config.n_recurrent_layers - 1):
        lstm = LSTM(32, kernel_regularizer=regularizers.l2(config.reg_weight))(lstm)
    rnn_out = lstm
    #rnn_out
    # to top layer of text network

    #concat them
    if config.img_only:
        x = cnn_out
    elif config.numeric_only:
        x = fc_out
    elif config.rnn_only:
        x = rnn_out
    else:
        x = concatenate([cnn_out, fc_out, rnn_out])

    predictions = Dense(config.n_classes, activation='softmax', name='main_output', kernel_regularizer=regularizers.l2(config.reg_weight))(x)

    model = Model(inputs=[numeric_inputs, image_model.input, text_inputs], outputs=predictions)

    if False:#weights is not None and config.freeze_cnn:
        for i in range(len(image_model.layers) - config.trainable_convnet_layers):
           image_model.layers[i].trainable = False

    def custom_loss(y_true, y_pred):
        main_loss = losses.sparse_categorical_crossentropy(y_true, y_pred)
        pred_indices = K.argmax(y_pred, axis=-1)
        distance_penalty = 1.0 / K.abs(pred_indices - config.n_classes / 2)
        return main_loss + config.distance_weight * distance_penalty

    opt = Adam(lr=config.lr)
    model.compile(optimizer=opt, loss=custom_loss, metrics=['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy'])

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
    model = load_keras_model(path + 'model')
    try:
        with open(path + 'config', 'rb') as pickle_file:
            config = pickle.load(pickle_file)
    except:
        config = None
    return model, config

