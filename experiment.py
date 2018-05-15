import numpy as np
from threading import Thread
from keras.models import Sequential
from sklearn import linear_model as sklm
import util
from model import build_model, Config, write_model, load_model
import os
import sys
import pickle
import argparse
import preprocessing

from keras.callbacks import ReduceLROnPlateau, TensorBoard, CSVLogger

num_data_files = 50
n_iterations = 1000

def train(args):
    if not os.path.exists('models/'):
        os.mkdir('models/')

    ## run param search and other stuff
    #x_train, y_train, x_dev, y_dev, x_test, y_test = util.load_for_lin_reg()
    #reg = linear_regression(x_train, y_train, x_dev, y_dev, x_test, y_test)

    numeric_data, text_data, prices = preprocessing.load_tabular_data()

    word_index, tokenizer = util.tokenize_texts(text_data)
    embedding_matrix = util.load_embedding_matrix(word_index)

    if args.folder is not None:
        config, model = load_model(args.folder)
        model_folder = 'models/' + args.folder + '/'
    else:
        config = Config(word_index, embedding_matrix, imagenet_weights=True, trainable_convnet_layers=20,
                    n_classes=1000)
        model = build_model(config)
        if os.path.exists('models/' + args.name):
            print('A folder with that name already exists.')
            exit()
        if args.name is not None:
            os.mkdir('models/' + args.name)
            model_folder = 'models/' + args.name + '/'
        else:
            model_subfolders = os.listdir('models/')
            model_folder = 'models/' + str(len(model_subfolders)) + '/'

    bins = util.get_bins(prices, num=config.n_classes)
    train_model(model, config, numeric_data, text_data, bins, model_folder)

'''
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
        train_model(model, config, numeric_data, text_data, ) '''


def logistic_regression(x_train, y_train, x_dev, y_dev, x_test, y_test):
    reg = sklm.logistic_regression_path()
    reg.fit(x_train, y_train)
    return reg

def train_model(model, config, numeric_data, text_data, bins, model_folder):
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

    #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
    #                              patience=4, min_lr=0.00001, cooldown=3)
    tensorboard = TensorBoard(log_dir=model_folder + 'logs/', write_images=True, histogram_freq=2)
    csvlogger = CSVLogger(model_folder + 'training_log.csv', append=True)

    history = None
    #training loop
    for iteration in range(n_iterations):
        print('Iteration: {}'.format(iteration))
        # start loading data
        data_thread = Thread(target=load_data_batch, args=(img_files, numeric_data, text_data, config.img_shape, False))
        data_thread.start()

        # fit model on data batch
        history = model.fit([numeric_data_batch[:, 1:3], img_data_batch],
                            util.buckets(numeric_data_batch[:, 3], bins, num=config.n_classes),
                            batch_size=config.batch_size, validation_split=0.1, epochs=1,
                            callbacks=[tensorboard, csvlogger])

        if history.history['val_loss'][-1] < best_val_loss:
            best_val_loss = history.history['val_loss'][-1]
            write_model(model, config, best_val_loss, model_folder)

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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains and tests the model.')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='trains model')
    command_parser.add_argument('-n', action='store', dest='name',
                                help="Save models to folder with designated name")
    command_parser.add_argument('-r', action='store', dest='folder', help="Resume training with existing model. Input a model folder name")
    command_parser.set_defaults(func=train)



    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)


