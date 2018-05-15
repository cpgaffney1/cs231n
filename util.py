import numpy as np
from PIL import Image
import os
import sys
import re
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


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


def buckets(x, num=1000):
    bins = np.linspace(0,np.max(x), num=num)
    y = np.digitize(x,bins, right=True)
    return y


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


def print_distribution(pred, real = None):
    bins = np.linspace(0, 200, 10)
    plt.hist(pred, bins, alpha=0.5, label='Prediction')
    if(real):
        plt.hist(real, bins, alpha=0.5, label='Real')
    plt.legend(loc='upper right')
    plt.title('Baseline Results')
    plt.show()
    if os.path.exists('Graphs/'):
        plt.savefig('Graphs/Baseline_Results')
    else:
        os.makedirs('Graphs/')
        plt.savefig('Graphs/Baseline_Results')
