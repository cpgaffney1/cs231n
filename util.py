import numpy as np
from PIL import Image
import os
import sys
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import preprocessing
import tensorflow as tf

from keras.applications.xception import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import ImageDataGenerator
from scipy.stats import kde

import keras.backend as K


def shuffle_in_parallel(arr1, arr2):
    assert(len(arr1) == len(arr2))
    indices = np.arange(len(arr1))
    np.random.shuffle(indices)
    return arr1[indices], arr2[indices]


#splits train, dev, test
def split_data(x, y):
    #train, dev, test
    split_portions = [0.8, 0.1, 0.1]
    assert(sum(split_portions) == 1)
    n_train_obs = int(split_portions[0] * len(x))
    n_dev_obs = int(split_portions[1] * len(x))
    n_test_obs = int(split_portions[2] * len(x))
    x_train = x[:n_train_obs]
    y_train = y[:n_train_obs]
    x_dev = x[n_train_obs: n_train_obs + n_dev_obs + 1]
    y_dev = y[n_train_obs: n_train_obs + n_dev_obs + 1]
    x_test = x[n_train_obs + n_dev_obs:]
    y_test = y[n_train_obs + n_dev_obs:]
    return x_train, y_train, x_dev, y_dev, x_test, y_test


def load_for_lin_reg():
    with open('tabular_data/scraped_data.csv') as f:
        lines = f.readlines()
        lines = lines[1:]
        X = np.zeros((len(lines), 2))
        y = np.zeros(len(lines))
        for i, line in enumerate(lines):
            sp = line.split(';,.')
            zpid, zip, price, beds, baths, descr, address = sp
            X[i][0] = float(beds)
            X[i][1] = float(baths)
            y[i] = float(price)

    X, y = shuffle_in_parallel(X, y)
    x_train, y_train, x_dev, y_dev, x_test, y_test = split_data(X, y)
    return x_train, y_train, x_dev, y_dev, x_test, y_test


def crop(image, shape=(299, 299), random=False):
    for i in range(len(shape)):
        assert(image.shape[i] >= shape[i])
    new_img = np.zeros((shape[0], shape[1], 3))
    if random:
        start = (np.random.randint(0, image.shape[0] - shape[0]), np.random.randint(0, image.shape[1] - shape[1]))
    else:
        start = ((image.shape[0] - shape[0])//2, (image.shape[1] - shape[1]) // 2)
    #new_img[:,:,:] = image[start[0]:start[0]+shape[0], start[1]:start[1]+shape[1], :]
    new_img = image[start[0]:shape[0]+start[0], start[1]:shape[1]+start[1], :]
    assert(new_img.shape[0] == shape[0] and new_img.shape[1] == shape[1])
    return new_img


def buckets(x, bins):
    y = np.digitize(x, bins, right=True)
    return y

def get_bins(prices, num=100):
    bins = np.geomspace(10000, 1e6, num=num)
    bins[-1] = np.max(prices)
    return bins


def clean_text(descr):
    '''if (descr[0] != '"' or descr[-1] != '"') and (descr[0] != "'" or descr[-1] != "'"):
        print(descr)'''
    assert((descr[0] == "'" and descr[-1] == "'") or (descr[0] == '"' and descr[-1] == '"'))
    descr = descr[1:-1]
    match = re.search(r'For sale: \$(\d*,)*\d*\.', descr)
    if match is None:
        print(descr)
    return descr[match.end():].strip()


def tokenize_texts(text_data):
    text_data_list = []
    for key in text_data.keys():
        descr = text_data[key][0]
        text_data_list.append(clean_text(descr))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_data_list)
    return tokenizer.word_index, tokenizer


def load_embedding_matrix(word_index, filename='glove.twitter.27B.50d.txt', embed_dim=50):
    embeddings_index = {}
    f = open('wordvec/' + filename, encoding='utf8', errors='replace')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((len(word_index) + 1, embed_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def print_history(history):
    if history is None:
        return
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    if os.path.exists('Graphs/'):
        plt.savefig('Graphs/train_history')
    else:
        os.makedirs('Graphs/')
        plt.savefig('Graphs/train_history')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    if os.path.exists('Graphs/'):
        plt.savefig('Graphs/train_history')
    else:
        os.makedirs('Graphs/')
        plt.savefig('Graphs/train_history')


def print_distribution(pred, bins, real=None):
    plt.hist(pred, bins, alpha=0.5, label='Prediction')
    if real is not None:
        plt.hist(real, bins, alpha=0.5, label='Real')
    plt.legend(loc='upper right')
    plt.title('Baseline Results')
    plt.show()
    if os.path.exists('Graphs/'):
        plt.savefig('Graphs/Baseline_Results')
    else:
        os.makedirs('Graphs/')
        plt.savefig('Graphs/Baseline_Results')

def save_file(x, name):
    out = csv.writer(open(name + '.csv', "w"), delimiter=',', quoting=csv.QUOTE_ALL)
    out.writecolumn(x)

'''def preprocess_numeric_data(num_data):
    zips = num_data[:, 0]
    prices = num_data[:, 3]

    preprocessed_num_data = np.zeros((num_data.shape[0], 2))
    # 0:zip, 1:beds, 2:baths
    preprocessed_num_data[:, :] = num_data[:, 1:3]
    # 3:average price
    #for i in range(preprocessed_num_data.shape[0]):
    #    preprocessed_num_data[i][3] = zips_to_avg_prices[preprocessed_num_data[i][0]]
    #exit()
    return preprocessed_num_data'''



def preprocess_numeric_data(num_data_orig, additional_data):
    if additional_data is not None:
        ZIP_COL = 0
        HPI_COL = -1
        additional_data = fill_missing_hpi(additional_data, ZIP_COL, HPI_COL)
        additional_data[:, 1:] = (additional_data[:, 1:] - np.mean(additional_data[:, 1:], axis=0)) / (
                np.std(additional_data[:, 1:], axis=0) + 0.000001)
        zip_to_additional_data = {}
        for i in range(additional_data.shape[0]):
            ######## ASSUME ZIP IS IN FIRST COLUMN !!!!!!!!!!!!!!!!!!!!!!!!!!
            zip_to_additional_data[additional_data[i][ZIP_COL]] = additional_data[i][1:]
    if additional_data is not None:
        num_features = additional_data.shape[1] + 2
    else:
        num_features = 2
    num_data = {}
    count = 0
    for zpid in num_data_orig.keys():
        zip, beds, baths, price = num_data_orig[zpid]
        data = np.zeros(num_features)
        data[0] = price
        data[1] = beds
        data[2] = baths
        if zip_to_additional_data is not None:
            try:
                data[3:] = zip_to_additional_data[int(zip)]
            except:
                count += 1
                continue
        num_data[zpid] = data
    return num_data

def fill_missing_hpi(num_data, zip_col, hpi_col):
    zip_to_hpi = {int(num_data[i][zip_col]): float(num_data[i][hpi_col]) for i in range(num_data.shape[0]) \
                  if not np.isnan(num_data[i][hpi_col])}
    zips_sorted = np.unique(sorted(zip_to_hpi.keys()))
    for i in range(num_data.shape[0]):
        if np.isnan(num_data[i][hpi_col]):
            index = np.searchsorted(zips_sorted, num_data[i][zip_col])
            if index == len(zips_sorted):
                index -= 1
            closest_zip = zips_sorted[index]
            num_data[i][hpi_col] = zip_to_hpi[int(closest_zip)]
    return num_data

def remove_price_array_from_numeric_data(num_data):
    return num_data[:, 0], num_data[:, 1:]

def load_data_batch(img_files, numeric_data, text_data, bins, img_shape,
                    verbose, batch_size, mode):
    img_data_batch, numeric_data_batch, descriptions_batch, addresses_batch = \
        preprocessing.process_data_batch(np.random.choice(img_files, size=batch_size, replace=False),
                                         text_data, numeric_data, desired_shape=img_shape, verbose=verbose, mode=mode)
    #img_data_batch = img_data_batch.astype(np.float32)
    #img_data_batch = preprocess_input(img_data_batch)
    y_batch, numeric_data_batch = remove_price_array_from_numeric_data(numeric_data_batch)
    return [numeric_data_batch, img_data_batch, descriptions_batch], buckets(y_batch, bins)

def generator(img_files, numeric_data, text_data, bins, img_shape=(299,299,3),
              verbose=False, batch_size=32, mode='train', tokenizer=None, maxlen=20,
              ):
    while True:
        x, y = load_data_batch(img_files, numeric_data, text_data, bins,
                              img_shape=img_shape, verbose=verbose, batch_size=batch_size, mode=mode)
        imgs = x[1]
        '''datagen = ImageDataGenerator(
            rotation_range=20,
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
        )
        datagen.fit(imgs)
        for imgs, y in datagen.flow(imgs, y, batch_size=batch_size):
            break'''
        sequences = np.asarray(tokenizer.texts_to_matrix(x[2]))
        sequences = pad_sequences(sequences, maxlen=maxlen)
        yield [x[0], imgs, sequences], y
        '''if img_only:
            yield imgs, y
        elif tokenizer is not None:
            sequences = np.asarray(tokenizer.texts_to_matrix(x[2]))
            sequences = pad_sequences(sequences, maxlen=maxlen)
            yield [x[0], imgs, sequences], y
        else:
            yield [x[0], imgs], y'''


def conf_matrix(y_true, y_false, nbins, suffix=''):
    # 2D Histogram
    plt.hist2d(y_true, y_false, bins=nbins, cmap=plt.cm.BuGn_r)
    plt.title('Confusion Matrix')
    if os.path.exists('Graphs/'):
        plt.savefig('Graphs/Confusion_Matrix' + suffix)
    else:
        os.makedirs('Graphs/')
        plt.savefig('Graphs/Confusion_Matrix' + suffix)

def save_saliency_imgs(img, suffix=''):
    plt.imsave('Graphs/saliency_map' + suffix, img)
