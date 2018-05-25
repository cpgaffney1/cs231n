import numpy as np
from threading import Thread
from model import build_model, Config, write_model, load_model
import os
import sys
import csv
from PIL import Image
import pickle


from keras.callbacks import ReduceLROnPlateau, TensorBoard, CSVLogger, ModelCheckpoint

from sklearn.linear_model import LogisticRegression

import argparse
import util

import preprocessing

num_data_files = 50
n_iterations = 1000
TRAIN_PHASE = 0
TEST_PHASE = 1

def baseline(args):
    n_classes = 100
    _, _, prices = preprocessing.load_tabular_data()
    bins = util.get_bins(prices, num=n_classes)
    x_train, y_train, x_dev, y_dev, x_test, y_test = util.load_for_lin_reg()

    y_train = util.buckets(y_train, bins)
    y_dev = util.buckets(y_dev, bins)
    y_test = util.buckets(y_test, bins)

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

    #K.clear_session()
    #K.set_learning_phase(TEST_PHASE)

    ## run param search and other stuff

    numeric_data, text_data, prices = preprocessing.load_tabular_data()

    word_index, tokenizer = util.tokenize_texts(text_data)
    embedding_matrix = util.load_embedding_matrix(word_index)

    if args.trainable_layers is None:
        trainable_convnet_layers = 10
    else:
        trainable_convnet_layers = int(args.trainable_layers)
    if args.reg_weight is None:
        reg_weight = 0.01
    else:
        reg_weight = float(args.reg_weight)

    if args.folder is not None:
        config, model = load_model(args.folder)
        model_folder = 'models/' + args.folder + '/'
    else:
        config = Config(word_index, embedding_matrix, imagenet_weights=True, trainable_convnet_layers=trainable_convnet_layers,
                    n_classes=1000, lr=0.0001, reg_weight=reg_weight)
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
    train_model(model, config, numeric_data, text_data, bins, model_folder, tokenizer)

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


def train_model(model, config, numeric_data, text_data, bins, model_folder, tokenizer):
    best_val_loss = float('inf')
    global loaded_img_data
    global loaded_numeric_data
    global loaded_descriptions

    train_img_files = os.listdir('imgs/')
    val_img_files = os.listdir('val_imgs/')

    #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
    #                              patience=4, min_lr=0.00001, cooldown=3)


    tensorboard = TensorBoard(log_dir=model_folder + 'logs/', write_images=True, write_grads=True)
    csvlogger = CSVLogger(model_folder + 'training_log.csv', append=True)
    saver = ModelCheckpoint(model_folder + 'model', monitor='val_sparse_categorical_accuracy', save_best_only=True, mode='max')

    history = model.fit_generator(util.generator(train_img_files, numeric_data, text_data, bins, img_shape=config.img_shape,
                                                 batch_size=config.batch_size, tokenizer=tokenizer, maxlen=config.max_seq_len),
                                  epochs=100, callbacks=[tensorboard, csvlogger, saver],
                                  validation_data=util.generator(
                                      val_img_files, numeric_data, text_data, bins, img_shape=config.img_shape, batch_size=config.batch_size,
                                      tokenizer=tokenizer, maxlen=config.max_seq_len, mode='val'
                                  ), steps_per_epoch=int(20000/config.batch_size), validation_steps=int(4500/config.batch_size))

def evaluate(args):
    config, model = load_model(args.name)

    val_img_files = os.listdir('val_imgs/')
    test_img_files = os.listdir('test_imgs/')
    numeric_data, text_data, prices = preprocessing.load_tabular_data()


    bins = util.get_bins(prices, num=config.n_classes)

    results = model.evaluate_generator(util.generator(
        val_img_files, numeric_data, text_data, bins, img_shape=config.img_shape,
        batch_size=config.batch_size, mode='val'), steps_per_epoch=int(4500/config.batch_size))
    print(results)

    results = model.evaluate_generator(util.generator(
        val_img_files, numeric_data, text_data, bins, img_shape=config.img_shape,
        batch_size=config.batch_size, mode='test'), steps_per_epoch=int(5000/config.batch_size))
    print(results)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains and tests the model.')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='trains model')
    command_parser.add_argument('-t', '--test', action='store_true', default=False, help="Run but save nothing. Used for testing code")
    command_parser.add_argument('-n', action='store', dest='name',
                                help="Save models to folder with designated name")
    command_parser.add_argument('-r', action='store', dest='folder', help="Resume training with existing model. Input a model folder name")
    command_parser.add_argument('-rw', action='store', dest='reg_weight',
                                help="Set reg weight param")
    command_parser.add_argument('-tl', action='store', dest='trainable_layers',
                                help="Set trainable layers params")
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


