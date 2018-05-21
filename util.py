import numpy as np
from PIL import Image
import os
import sys
import re
import matplotlib.pyplot as plt
import csv
import preprocessing

from keras.applications.xception import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import ImageDataGenerator



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

def get_bins(prices, num=1000):
    bins = np.geomspace(10000, max(prices), num=num)
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

def load_data_batch(img_files, numeric_data, text_data, bins, img_shape,
                    verbose, batch_size, mode):
    img_data_batch, numeric_data_batch, descriptions_batch, addresses_batch = \
        preprocessing.process_data_batch(np.random.choice(img_files, size=batch_size, replace=False),
                                         text_data, numeric_data, desired_shape=img_shape, verbose=verbose, mode=mode)
    img_data_batch = img_data_batch.astype(np.float32)
    img_data_batch = preprocess_input(img_data_batch)
    return [numeric_data_batch[:, 1:3], img_data_batch], buckets(numeric_data_batch[:, 3], bins)

def generator(img_files, numeric_data, text_data, bins, img_shape=(299,299,3),
              verbose=False, batch_size=32, mode='train'):
    while True:
        x, y = load_data_batch(img_files, numeric_data, text_data, bins,
                              img_shape=img_shape, verbose=verbose, batch_size=batch_size, mode=mode)
        imgs = x[1]
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            horizontal_flip=True)
        datagen.fit(imgs)
        for i in range(1):
            ret = datagen.flow(imgs, y, batch_size=batch_size)
            print(ret)
        yield [x[0], imgs], y