import numpy as np
from threading import Thread
from model import build_model, Config, write_model, load_model
import os
import sys
import csv
from PIL import Image
import pickle
import keras.backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras.losses as losses
from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import ReduceLROnPlateau, TensorBoard, CSVLogger, ModelCheckpoint

from sklearn.linear_model import LogisticRegression

import argparse
import util

import preprocessing

num_data_files = 50
n_iterations = 1000
TRAIN_PHASE = 0
TEST_PHASE = 1

def visualize(args):
    img_files = os.listdir('imgs/')
    n_classes = 50
    img_shape = (224,224,3)
    numeric_data, text_data, prices = preprocessing.load_tabular_data()
    bins = util.get_bins(prices, num=n_classes)

    additional_num_data = np.load('tabular_data/add_num_data.npy')
    numeric_data = util.preprocess_numeric_data(numeric_data, additional_num_data)

    x, y = util.load_data_batch(img_files, numeric_data, text_data, bins, img_shape,
                                True, len(img_files), 'train')
    x=x[1]

    found_indices = [-1, -1, -1, -1, -1]
    index = len(x)-1
    while(np.min(found_indices) == -1):
        if index % 100 == 0:
            print(found_indices)
        if y[index] < 10:
            found_indices[0] = index
        elif y[index] < 20:
            found_indices[1] = index
        elif y[index] < 30:
            found_indices[2] = index
        elif y[index] < 40:
            found_indices[3] = index
        else:
            found_indices[4] = index
        index -= 1

    found_indices = np.asarray(found_indices, dtype='int32')
    print(found_indices)
    img_list = x[found_indices]
    labels = y[found_indices]

    img_arr = np.concatenate(img_list)
    merged_img = Image.fromarray(img_arr)

    merged_img.save('merged_images.jpg')

    print('Buckets:')
    print(labels)


def baseline(args):
    n_classes = 50
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
    np.savetxt('train_actual_linear.csv', y_train, delimiter=',')
    np.savetxt('bins.csv', bins, delimiter=',')

    if reg is None:
        print('Beginning regression')
        reg = LogisticRegression(verbose=10, multi_class='multinomial', solver='saga')
        reg.fit(x_train, y_train)
        with open('linear_model', 'wb') as pickle_file:
            pickle.dump(reg, pickle_file)
    train_pred = reg.predict(x_train)
    dev_pred = reg.predict(x_dev)

    print('Train scores')
    print(reg.score(x_train, y_train))
    print('Validation scores')
    print(reg.score(x_dev, y_dev))
    print('Test scores')
    print(reg.score(x_test, y_test))
    np.savetxt('train_preds_linear.csv', train_pred, delimiter=',')
    util.conf_matrix(y_train, train_pred, 100, suffix='_' + 'linear')

    util.print_distribution(train_pred, bins, real=y_train)
    util.print_distribution(dev_pred, bins, real=y_dev)


def train(args):
    if not os.path.exists('models/'):
        os.mkdir('models/')

    #numeric data is a map of zpid to tuple of (zip, beds, baths, price)
    numeric_data, text_data, prices = preprocessing.load_tabular_data()

    word_index, tokenizer = util.tokenize_texts(text_data)
    embedding_matrix = util.load_embedding_matrix(word_index)
    additional_num_data = np.load('tabular_data/add_num_data.npy')

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
                    n_classes=50, lr=0.0001, reg_weight=reg_weight, img_only=args.img_only, numeric_input_size=additional_num_data.shape[1]+2-1,
                        numeric_only=args.numeric_only, distance_weight=10)
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

    numeric_data = util.preprocess_numeric_data(numeric_data, additional_num_data)
    bins = util.get_bins(prices, num=config.n_classes)
    train_model(model, config, numeric_data, text_data, bins, model_folder, tokenizer, args.overfit)


def train_model(model, config, numeric_data, text_data, bins, model_folder, tokenizer, overfit):
    train_img_files = os.listdir('imgs/')
    val_img_files = os.listdir('val_imgs/')

    if overfit:
        np.random.shuffle(train_img_files)
        np.random.shuffle(val_img_files)
        train_img_files = train_img_files[:128]
        val_img_files = val_img_files[:128]

    tensorboard = TensorBoard(log_dir=model_folder + 'logs/', write_images=True, write_grads=True)
    csvlogger = CSVLogger(model_folder + 'training_log.csv', append=True)
    saver = ModelCheckpoint(model_folder + 'model', monitor='val_sparse_categorical_accuracy', save_best_only=True, mode='max')

    with open(model_folder + 'config', 'wb') as pickle_file:
        pickle.dump(config, pickle_file)
    history = model.fit_generator(util.generator(train_img_files, numeric_data, text_data, bins, img_shape=config.img_shape,
                                                 batch_size=config.batch_size, tokenizer=tokenizer, maxlen=config.max_seq_len),
                                  epochs=100, callbacks=[tensorboard, csvlogger, saver],
                                  validation_data=util.generator(
                                      val_img_files, numeric_data, text_data, bins,
                                      img_shape=config.img_shape, batch_size=config.batch_size,
                                      tokenizer=tokenizer, maxlen=config.max_seq_len, mode='val'
                                  ), steps_per_epoch=int(len(train_img_files)/8/config.batch_size), validation_steps=int(len(val_img_files)/2/config.batch_size))


def evaluate(args):
    def custom_loss(y_true, y_pred):
        epsilon = 0.001
        main_loss = losses.sparse_categorical_crossentropy(y_true, y_pred)
        pred_indices = K.argmax(y_pred, axis=-1)
        pred_indices = K.cast(pred_indices, dtype='float32')
        distance_penalty = K.constant(1.0, dtype='float32') / (K.abs(pred_indices - K.constant(config.n_classes / 2.0, dtype='float32')) + epsilon)
        return main_loss + config.distance_weight * distance_penalty
    model, config = load_model(args.name)
    if args.test:
        mode = 'test'
    else:
        mode = 'val'

    input_type = util.get_input_type(config)

    img_files = os.listdir(mode + '_imgs/')
    numeric_data, text_data, prices = preprocessing.load_tabular_data()
    additional_num_data = np.load('tabular_data/add_num_data.npy')
    numeric_data = util.preprocess_numeric_data(numeric_data, additional_num_data)

    bins = util.get_bins(prices, num=config.n_classes)
    print('Beginning evaluation...')
    results = model.evaluate_generator(util.generator(
        img_files, numeric_data, text_data, bins, img_shape=config.img_shape,
        batch_size=config.batch_size, mode=mode,
        tokenizer=config.tokenizer, maxlen=config.max_seq_len,
        input_type=input_type, eval=True), steps=int(len(img_files)/config.batch_size)
    )
    print(results)



def show_saliency(args):
    def custom_loss(y_true, y_pred):
        epsilon = 0.001
        main_loss = losses.sparse_categorical_crossentropy(y_true, y_pred)
        pred_indices = K.argmax(y_pred, axis=-1)
        pred_indices = K.cast(pred_indices, dtype='float32')
        distance_penalty = K.constant(1.0, dtype='float32') / (K.abs(pred_indices - K.constant(config.n_classes / 2.0, dtype='float32')) + epsilon)
        return main_loss + config.distance_weight * distance_penalty
    model, config = load_model(args.name)
    input_type = util.get_input_type(config)

    numeric_data, text_data, prices = preprocessing.load_tabular_data()
    additional_num_data = np.load('tabular_data/add_num_data.npy')
    numeric_data = util.preprocess_numeric_data(numeric_data, additional_num_data)
    bins = util.get_bins(prices, num=config.n_classes)

    img_files = os.listdir('imgs/')
    np.random.shuffle(img_files)
    number = 256
    x, y = util.load_data_batch(img_files[:number], numeric_data, text_data, bins, config.img_shape,
                                False, number, 'train')
    sequences = np.asarray(config.tokenizer.texts_to_matrix(x[2]))
    sequences = pad_sequences(sequences, maxlen=config.max_seq_len)
    x[2] = sequences

    if input_type == 'full':
        x = [x[0], x[1], sequences]
    elif input_type == 'img':
        x = x[1]
    elif input_type == 'num':
        x = x[0]
    elif input_type == 'rnn':
        x = sequences
    else:
        print('error')
        exit()

    _ = model.predict(x)
    print('Visualizing saliency...')

    folder = 'models/' + args.name + '/'

    indices = np.arange(0, x[0].shape[0])
    np.random.shuffle(indices)
    indices = indices[:64]
    x = [x[0][indices], x[1][indices], x[2][indices]]
    y = y[indices]

    label_tensor = K.constant(y)
    fn = K.function(model.inputs,
                K.gradients(losses.sparse_categorical_crossentropy(label_tensor, model.outputs[0]),
                                    model.inputs))
    grads = fn([x])
    grads = grads[0]

    saliency = np.absolute(grads).max(axis=-1)

    merged = np.concatenate((saliency[i] for i in range(saliency.shape[0])), axis=0)
    plt.imsave(folder + 'saliency.jpg', merged, cmap=plt.cm.hot)


def pred(args):
    def custom_loss(y_true, y_pred):
        epsilon = 0.001
        main_loss = losses.sparse_categorical_crossentropy(y_true, y_pred)
        pred_indices = K.argmax(y_pred, axis=-1)
        pred_indices = K.cast(pred_indices, dtype='float32')
        distance_penalty = K.constant(1.0, dtype='float32') / (K.abs(pred_indices - K.constant(config.n_classes / 2.0, dtype='float32')) + epsilon)
        return main_loss + config.distance_weight * distance_penalty
    model, config = load_model(args.name)
    if args.test:
        mode = 'test'
    else:
        mode = 'val'

    input_type = util.get_input_type(config)

    numeric_data, text_data, prices = preprocessing.load_tabular_data()
    additional_num_data = np.load('tabular_data/add_num_data.npy')
    numeric_data = util.preprocess_numeric_data(numeric_data, additional_num_data)
    bins = util.get_bins(prices, num=config.n_classes)

    img_files = os.listdir(mode + '_imgs/')
    np.random.shuffle(img_files)

    x, y = util.load_data_batch(img_files, numeric_data, text_data, bins, config.img_shape,
                                True, len(img_files), mode)

    sequences = np.asarray(config.tokenizer.texts_to_matrix(x[2]))
    sequences = pad_sequences(sequences, maxlen=config.max_seq_len)
    x[2] = sequences

    if input_type == 'full':
        x = [x[0], x[1], sequences]
    elif input_type == 'img':
        x = x[1]
    elif input_type == 'num':
        x = x[0]
    elif input_type == 'rnn':
        x = sequences
    else:
        print('error')
        exit()

    predictions = model.predict(x)

    folder = 'models/' + args.name + '/'

    print('Writing confusion matrix...')
    print(np.argmax(predictions, axis=-1).shape)
    np.savetxt(folder + 'preds_neural.csv', np.argmax(predictions, axis=-1), delimiter=',')
    util.conf_matrix(y, np.argmax(predictions, axis=-1), config.n_classes, folder)


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
    command_parser.add_argument('-i', '--img_only', action='store_true', default=False, help="Only use img input")
    command_parser.add_argument('-num', '--numeric_only', action='store_true', default=False, help="Only use numeric input")
    command_parser.add_argument('-rnn', '--rnn_only', action='store_true', default=False, help="Only use rnn input")
    command_parser.add_argument('-o', '--overfit', action='store_true', default=False, help="Try to overfit on small dataset")


    command_parser.set_defaults(func=train)

    command_parser = subparsers.add_parser('base', help='trains baseline model')
    command_parser.add_argument('-r', '--resume', action='store_true', default=False, help="Resume")
    command_parser.set_defaults(func=baseline)

    command_parser = subparsers.add_parser('eval', help='evaluate model')
    command_parser.add_argument('-n', action='store', dest='name',
                                help="load model with selected name")
    command_parser.add_argument('-t', '--test', action='store_true', default=False, help="Do on test set. default is validation set")
    command_parser.set_defaults(func=evaluate)

    command_parser = subparsers.add_parser('vis', help='visualize iages')
    command_parser.set_defaults(func=visualize)

    command_parser = subparsers.add_parser('sal', help='do saliency')
    command_parser.add_argument('-n', action='store', dest='name',
                                help="load model with selected name")
    command_parser.add_argument('-t', '--test', action='store_true', default=False,
                                help="Do on test set. default is validation set")
    command_parser.set_defaults(func=show_saliency)

    command_parser = subparsers.add_parser('pred', help='make predictions and save')
    command_parser.add_argument('-n', action='store', dest='name',
                                help="load model with selected name")
    command_parser.add_argument('-t', '--test', action='store_true', default=False,
                                help="Do on test set. default is validation set")
    command_parser.set_defaults(func=pred)


    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)


