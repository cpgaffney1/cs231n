import numpy as np
from threading import Thread
from model import build_model, Config, write_model, load_model
import os
import sys
import csv
from PIL import Image
import pickle
import keras.backend as K


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
    np.savetxt('bins.csv', bins, delimiter=',')

    util.conf_matrix(y_train, train_pred, 100, suffix='_' + 'linear')

    util.print_distribution(train_pred, bins, real=y_train)
    util.print_distribution(dev_pred, bins, real=y_dev)


def train(args):
    if not os.path.exists('models/'):
        os.mkdir('models/')

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
        model, config = load_model(args.folder)
        model_folder = 'models/' + args.folder + '/'
    else:
        config = Config(word_index, embedding_matrix, tokenizer, imagenet_weights=True, trainable_convnet_layers=trainable_convnet_layers,
                    n_classes=100, lr=0.0001, reg_weight=reg_weight)
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

    additional_num_data = np.genfromtxt('FILENAME', skip_header=1, missing_values=[], filling_values=[np.nan], )
    numeric_data = util.preprocess_numeric_data(numeric_data, additional_num_data, num_features=config.numeric_input_size)
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

    with open(model_folder + 'config', 'wb') as pickle_file:
        pickle.dump(config, pickle_file)
    history = model.fit_generator(util.generator(train_img_files, numeric_data, text_data, bins, img_shape=config.img_shape,
                                                 batch_size=config.batch_size, tokenizer=tokenizer, maxlen=config.max_seq_len),
                                  epochs=100, callbacks=[tensorboard, csvlogger, saver],
                                  validation_data=util.generator(
                                      val_img_files, numeric_data, text_data, bins, img_shape=config.img_shape, batch_size=config.batch_size,
                                      tokenizer=tokenizer, maxlen=config.max_seq_len, mode='val'
                                  ), steps_per_epoch=int(20000/config.batch_size), validation_steps=int(len(val_img_files)/config.batch_size))

def evaluate(args):
    model, config = load_model(args.name)
    if args.test:
        mode = 'test'
    else:
        mode = 'val'

    img_files = os.listdir(mode + '_imgs/')
    numeric_data, text_data, prices = preprocessing.load_tabular_data()

    if config is None:
        img_only_model = True
        word_index, tokenizer = util.tokenize_texts(text_data)
        embedding_matrix = util.load_embedding_matrix(word_index)
        config = Config(word_index, embedding_matrix, tokenizer, imagenet_weights=True,
                    trainable_convnet_layers=10,
                    n_classes=100, lr=0.0001, reg_weight=0.01)
    else:
        img_only_model = False

    bins = util.get_bins(prices, num=config.n_classes)

    results = model.evaluate_generator(util.generator(
        img_files, numeric_data, text_data, bins, img_shape=config.img_shape,
        batch_size=config.batch_size, mode=mode,
        tokenizer=config.tokenizer, maxlen=config.max_seq_len, img_only=img_only_model), steps=int(256/config.batch_size)
    )
    print(results)

    x, y = util.load_data_batch(img_files, numeric_data, text_data, bins, config.img_shape,
                    False, len(img_files), mode)
    x = x[:256]
    y = y[:256]
    if img_only_model:
        x = x[1]
    predictions = model.predict(x)

    util.conf_matrix(y, predictions, config.n_classes, suffix='_' + mode)
    np.savetxt('bins.csv', bins, delimiter=',')
    np.savetxt('train_preds_CNN.csv', predictions, delimiter=',')

    vis_indices = list(range(10))

    if img_only_model:
        vis_x = x
    else:
        vis_x = [x[0][vis_indices], x[1][vis_indices], x[2][vis_indices]]
    vis_y = y[vis_indices]
    vis_predictions = model.predict(vis_x)

    label_tensor = K.constant(vis_y)
    if img_only_model:
        fn = K.function(model.inputs, K.gradients(model.loss(label_tensor, model.outputs), model.inputs))
    else:
        fn = K.function(model.inputs, K.gradients(model.loss(label_tensor, model.outputs), model.inputs[1]))
    grads = K.eval(fn(vis_x))

    saliency = np.absolute(grads).max(axis=-1)
    print(saliency)


'''
    show_saliency(model, mode)


def show_saliency(model, mode):
    layer_idx = vis_utils.find_layer_idx(model, 'main_output')
    input_idx = vis_utils.find_layer_idx(model, 'img_input')
    input_tensor = model.layers[input_idx]
    model.layers[layer_idx].activation = activations.linear
    model = vis_utils.apply_modifications(model)
    filter_indices = [10, 30, 50, 70, 90]
    for f_idx in filter_indices:
        img = visualize_activation(model, layer_idx, filter_indices=f_idx, seed_input=input_tensor)
        util.save_saliency_imgs(img, suffix='_' + mode + '_{}'.format(f_idx))
'''

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
    command_parser.add_argument('-t', '--test', action='store_true', default=False, help="Do on test set. default is validation set")
    command_parser.set_defaults(func=evaluate)


    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)


