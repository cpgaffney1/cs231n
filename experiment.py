import numpy as np
from threading import Thread
from model import build_model, Config, write_model, load_model
import os
import sys
import keras.backend as K
import csv
from PIL import Image
from keras.optimizers import Adam
import pickle
from keras.models import load_model as load_keras_model


from keras.callbacks import ReduceLROnPlateau, TensorBoard, CSVLogger

from sklearn.linear_model import LogisticRegression

import argparse
import util

import preprocessing

num_data_files = 50
n_iterations = 1000
TRAIN_PHASE = 0
TEST_PHASE = 1

def baseline(args):
    n_classes = 1000
    _, _, prices = preprocessing.load_tabular_data()
    bins = util.get_bins(prices, num=n_classes)
    x_train, y_train, x_dev, y_dev, x_test, y_test = util.load_for_lin_reg()

    y_train = util.buckets(y_train, bins, num=n_classes)
    y_dev = util.buckets(y_dev, bins, num=n_classes)
    y_test = util.buckets(y_test, bins, num=n_classes)

    reg = None
    if args.resume:
        with open('linear_model', 'rb') as pickle_file:
            reg = pickle.load(pickle_file)

    logistic_regression(bins, x_train, y_train, x_dev, y_dev, x_test, y_test, reg=reg)

def logistic_regression(bins, x_train, y_train, x_dev, y_dev, x_test, y_test, reg=None):
    if reg is None:
        print('Beginning regression')
        reg = LogisticRegression(verbose=10, multi_class='multinomial', solver='saga')
        reg.fit(x_train, y_train)
        with open('linear_model', 'wb') as pickle_file:
            pickle.dump(reg, pickle_file)
    train_pred = reg.predict(x_train)
    dev_pred = reg.predict(x_dev)

    print(reg.score(x_train, y_train))
    print(reg.score(x_dev, y_dev))

    np.savetxt('train_preds_linear.csv', train_pred, delimiter=',')
    np.savetxt('train_actual_linear.csv', y_train, delimiter=',')
    exit()

    util.print_distribution(train_pred, bins, real=y_train)
    util.print_distribution(dev_pred, bins, real=y_dev)


def train(args):
    if not os.path.exists('models/'):
        os.mkdir('models/')

    K.clear_session()
    K.set_learning_phase(TEST_PHASE)

    ## run param search and other stuff

    numeric_data, text_data, prices = preprocessing.load_tabular_data()

    word_index, tokenizer = util.tokenize_texts(text_data)
    embedding_matrix = util.load_embedding_matrix(word_index)

    if args.folder is not None:
        config, model = load_model(args.folder)
        model_folder = 'models/' + args.folder + '/'
    else:
        config = Config(word_index, embedding_matrix, imagenet_weights=True, trainable_convnet_layers=20,
                    n_classes=1000, lr=0.01)
        model = build_model(config)
        if args.name is not None:
            if os.path.exists('models/' + args.name):
                print('A folder with that name already exists.')
                exit()
            os.mkdir('models/' + args.name)
            model_folder = 'models/' + args.name + '/'
        else:
            if not args.test:
                model_subfolders = os.listdir('models/')
                model_folder = 'models/' + str(len(model_subfolders)) + '/'
            else:
                model_folder = ''

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
    #print(util.buckets(numeric_data_batch[:, 3], bins, num=config.n_classes))
    #

    tensorboard = TensorBoard(log_dir=model_folder + 'logs/', write_images=True)
    csvlogger = CSVLogger(model_folder + 'training_log.csv', append=True)

    history = None
    #training loop
    for iteration in range(n_iterations):
        print('Iteration: {}'.format(iteration))
        # start loading data
        data_thread = Thread(target=load_data_batch, args=(img_files, numeric_data, text_data, config.img_shape, False))
        data_thread.start()

        # fit model on data batch
        validation_cutoff = int(0.9 * len(img_data_batch))
        history = model.fit([numeric_data_batch[:validation_cutoff, 1:3], img_data_batch[:validation_cutoff]],
                            util.buckets(numeric_data_batch[:validation_cutoff, 3], bins, num=config.n_classes),
                            batch_size=config.batch_size, epochs=1,
                            callbacks=[tensorboard, csvlogger])

        results = model.evaluate([numeric_data_batch[validation_cutoff:, 1:3], img_data_batch[validation_cutoff:]],
                                 util.buckets(numeric_data_batch[validation_cutoff:, 3], bins, num=config.n_classes).astype(int),
                                 batch_size=config.batch_size)
        print(results)


        if results[1] < best_val_loss:
            best_val_loss = results[1]
            write_model(model, config, best_val_loss, model_folder)

        # retrieve new data
        data_thread.join()
        img_data_batch = loaded_img_data.copy()
        numeric_data_batch = loaded_numeric_data.copy()
        text_data_batch = loaded_descriptions.copy()


    util.print_history(history)


def evaluate(args):
    config, _ = load_model(args.name)

    img_files = os.listdir('imgs/')
    numeric_data, text_data, prices = preprocessing.load_tabular_data()
    load_data_batch(img_files, numeric_data, text_data, img_shape=config.img_shape, batch_size=500)
    img_data_batch = loaded_img_data.copy()
    numeric_data_batch = loaded_numeric_data.copy()
    text_data_batch = loaded_descriptions.copy()

    bins = util.get_bins(prices, num=config.n_classes)

    K.clear_session()
    K.set_learning_phase(TEST_PHASE)
    config, model = load_model(args.name)
    # The accuracy will be close to the one of the training set on the last iteration.
    results = model.evaluate([numeric_data_batch[:, 1:3], img_data_batch],
                             util.buckets(numeric_data_batch[:, 3], bins, num=config.n_classes).astype(int),
                             batch_size=config.batch_size)
    print(results)



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
    command_parser.add_argument('-t', '--test', action='store_true', default=False, help="Run but save nothing. Used for testing code")
    command_parser.add_argument('-n', action='store', dest='name',
                                help="Save models to folder with designated name")
    command_parser.add_argument('-r', action='store', dest='folder', help="Resume training with existing model. Input a model folder name")
    command_parser.set_defaults(func=train)

    command_parser = subparsers.add_parser('base', help='trains baseline model')
    command_parser.add_argument('-r', '--resume', action='store_true', default=False, help="Resume")
    command_parser.set_defaults(func=baseline)

    command_parser = subparsers.add_parser('eval', help='evaluate model')
    command_parser.add_argument('-n', action='store', dest='name',
                                help="load model with selected name")
    command_parser.set_defaults(func=evaluate)


    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)


