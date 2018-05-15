import numpy as np
from threading import Thread
from keras.models import Sequential
from sklearn import linear_model as sklm
import util
from model import build_model, Config, write_model
import os
import sys
import pickle
import preprocessing

from keras.callbacks import ReduceLROnPlateau, TensorBoard

num_data_files = 50
n_epochs = 100

def main():
    print()
    if not os.path.exists('models/'):
        os.mkdir('models/')
    ## run param search and other stuff
    #x_train, y_train, x_dev, y_dev, x_test, y_test = util.load_for_lin_reg()
    #reg = linear_regression(x_train, y_train, x_dev, y_dev, x_test, y_test)

    numeric_data, text_data = preprocessing.load_tabular_data()

    word_index, tokenizer = util.tokenize_texts(text_data)
    embedding_matrix = util.load_embedding_matrix(word_index)

    config = Config(word_index, embedding_matrix, imagenet_weights=True, trainable_convnet_layers=20,
                    n_classes=500)
    model = build_model(config)
    train_model(model, config, numeric_data, text_data)

def sample_params():
    lr = np.random.uniform(0.000001, 0.01)
    n_recurrent_layers = np.random.randint(1, 3)
    n_numeric_layers = np.random.randint(1, 3)
    n_convnet_fc_layers = np.random.randint(1, 4)
    trainable_convnet_layers = np.random.randint(0, 30)
    imagenet_weights = np.random.randint(0, 1)
    n_top_hidden_layers = np.random.randint(1, 5)
    drop_prob = 0.0
    return lr, n_recurrent_layers, n_numeric_layers, trainable_convnet_layers, imagenet_weights, \
        n_top_hidden_layers, n_convnet_fc_layers, drop_prob


def optimize_params(word_index, embedding_matrix, n_trials=1000):
    numeric_data, text_data = preprocessing.load_tabular_data()
    for t in range(n_trials):
        lr, n_recurrent_layers, n_numeric_layers, trainable_convnet_layers, imagenet_weights, \
            n_top_hidden_layers, n_convnet_fc_layers, drop_prob = sample_params()
        config = Config(word_index, embedding_matrix, lr=lr, n_recurrent_layers=n_recurrent_layers,
                        n_numeric_layers=n_numeric_layers, trainable_convnet_layers=trainable_convnet_layers,
                        imagenet_weights=imagenet_weights,
                        n_top_hidden_layers= n_top_hidden_layers, n_convnet_fc_layers=n_convnet_fc_layers,
                        drop_prob=drop_prob)
        model = build_model(config)
        train_model(model, config, numeric_data, text_data)


def linear_regression(x_train, y_train, x_dev, y_dev, x_test, y_test):
    reg = sklm.LinearRegression()
    reg.fit(x_train, y_train)
    return reg

def train_model(model, config, numeric_data, text_data):
    best_val_loss = float('inf')
    global loaded_img_data
    global loaded_numeric_data
    global loaded_descriptions

    img_files = os.listdir('imgs/')

    #load initial data
    load_data_batch(img_files, numeric_data, text_data, img_shape=config.img_shape)
    img_data_batch = loaded_img_data.copy()
    numeric_data_batch = loaded_numeric_data.copy()
    text_data_batch = loaded_descriptions.copy()

    history = None
    #training loop
    for epoch in range(n_epochs):
        print('Fitting, epoch: {}'.format(epoch))
        # start loading data
        data_thread = Thread(target=load_data_batch, args=(img_files, numeric_data, text_data, config.img_shape, False))
        data_thread.start()

        # fit model on data batch
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                                      patience=3, min_lr=0.0000001)
        tensorboard = TensorBoard(log_dir='logs/')
        history = model.fit([numeric_data_batch[:, 1:3], img_data_batch],
                            util.buckets(numeric_data_batch[:, 3], num=config.n_classes),
                            batch_size=config.batch_size, validation_split=0.1, epochs=3,
                            callbacks=[reduce_lr, tensorboard])

        if history.history['val_loss'] < best_val_loss:
            best_val_loss = history.history['val_loss']
            write_model(model, config, best_val_loss)

        # retrieve new data
        data_thread.join()
        img_data_batch = loaded_img_data.copy()
        numeric_data_batch = loaded_numeric_data.copy()
        text_data_batch = loaded_descriptions.copy()


    util.print_history(history)


loaded_img_data = None
loaded_numeric_data = None
loaded_descriptions = None
def load_data_batch(img_files, numeric_data, text_data, img_shape=(299,299,3), verbose=True, batch_size=5000):
    global loaded_img_data
    global loaded_numeric_data
    global loaded_descriptions
    loaded_img_data, loaded_numeric_data, loaded_descriptions, addresses = \
        preprocessing.process_data_batch(np.random.choice(img_files, size=batch_size, replace=False), text_data, numeric_data,
                                         desired_shape=img_shape, verbose=verbose)
    #loaded_img_data = np.load('data/img_data{}.npy'.format(index))
    #loaded_numeric_data = np.load('data/numeric_data{}.npy'.format(index))




if __name__ == "__main__":
    main()

