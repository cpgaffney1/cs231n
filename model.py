from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, concatenate, LSTM
from keras.models import Model
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.optimizers import Adam
from keras.layers import Embedding

class Config:
    numeric_input_size = 2
    img_shape = (224,224,3)
    n_classes = 1000
    batch_size = 64
    embed_dim = 50
    max_seq_len = 30
    def __init__(self, word_index, embedding_matrix):
        self.word_index = word_index
        self.embedding_matrix = embedding_matrix
        self.vocab_size = len(word_index)
        self.embed_dim = embedding_matrix.shape[1]
        self.text_shape = (self.max_seq_len, self.vocab_size)

def build_model(config):
    numeric_inputs = Input(shape=(config.numeric_input_size,))
    img_inputs = Input(shape=config.img_shape)
    text_inputs = Input(shape=config.text_shape)

    #running cnn
    #image_model = Xception(include_top=False, weights='imagenet', input_tensor=img_inputs, input_shape=config.img_shape,
    #                                     pooling=None, classes=config.n_classes)
    image_model = MobileNet(input_shape=config.img_shape, include_top=False, weights='imagenet',
                        input_tensor=img_inputs, classes=config.n_classes)
    #freeze lower layers
    for layer in image_model.layers:
        layer.trainable = True

    cnn_out = image_model(img_inputs)
    cnn_out = Flatten()(cnn_out)
    x = Dense(128, activation='relu')(cnn_out)
    x = Dense(64, activation='relu')(x)
    cnn_out = Dense(64, activation='relu')(x)

    #running fc
    x = Dense(64, activation='relu')(numeric_inputs)
    fc_out = Dense(64, activation='relu')(x)

    #running RNN
    embedding_layer = Embedding(len(config.word_index) + 1, config.embed_dim,
                                weights=[config.embedding_matrix],
                                input_length=config.max_seq_len,
                                trainable=False)
    embedded_seqs = embedding_layer(text_inputs)
    lstm = LSTM(128)(embedded_seqs)
    rnn_out = LSTM(64)(lstm)
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
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy'])
    return model
