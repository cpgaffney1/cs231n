import numpy as np
from threading import Thread
from keras.models import Sequential
from sklearn import linear_model as sklm
import util
from model import build_model, Config
import os
import preprocessing

from keras.callbacks import ReduceLROnPlateau

num_data_files = 50
n_epochs = 10

def main():
    print()
    ## run param search and other stuff
    #x_train, y_train, x_dev, y_dev, x_test, y_test = util.load_for_lin_reg()
    #reg = linear_regression(x_train, y_train, x_dev, y_dev, x_test, y_test)

    numeric_data, text_data = preprocessing.load_tabular_data()

    word_index, tokenizer = util.tokenize_texts(text_data)
    embedding_matrix = util.load_embedding_matrix(word_index)

    config = Config(word_index, embedding_matrix)
    model = build_model(config)
    train_model(model, config, numeric_data, text_data)


def linear_regression(x_train, y_train, x_dev, y_dev, x_test, y_test):
    reg = sklm.LinearRegression()
    reg.fit(x_train, y_train)
    return reg

def train_model(model, config, numeric_data, text_data):
    global loaded_img_data
    global loaded_numeric_data
    global loaded_descriptions

    img_files = os.listdir('imgs/')

    #load initial data
    load_data_batch(img_files, numeric_data, text_data, img_shape=config.img_shape)
    img_data_batch = loaded_img_data.copy()
    numeric_data_batch = loaded_numeric_data.copy()
    text_data_batch = loaded_descriptions.copy()

    data_indices = np.asarray(list(range(num_data_files)))
    np.random.shuffle(data_indices)
    #training loop
    for epoch in range(n_epochs):
        for index in data_indices:
            print('Fitting, epoch: {}'.format(epoch))
            #start loading data
            #data_thread = Thread(target=load_data_batch, args=(img_files, numeric_data, text_data, img_))
            #data_thread.start()

            # fit model on data batch
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=3, min_lr=0.0000001)
            model.fit([numeric_data_batch[:100, 1:3], img_data_batch[:100, :, :, :]],
                      util.buckets(numeric_data_batch[:200, 3], num=config.n_classes),
                      batch_size=config.batch_size, validation_split=0.1, epochs=1000,
                      callbacks=[reduce_lr])

            #retrieve new data
            #data_thread.join()
            #img_data = loaded_img_data.copy()
            #numeric_data = loaded_numeric_data.copy()


loaded_img_data = None
loaded_numeric_data = None
loaded_descriptions = None
def load_data_batch(img_files, numeric_data, text_data, img_shape=(299,299,3), batch_size=1000):
    global loaded_img_data
    global loaded_numeric_data
    global loaded_descriptions
    loaded_img_data, loaded_numeric_data, loaded_descriptions, addresses = \
        preprocessing.process_data_batch(np.random.choice(img_files, size=batch_size), text_data, numeric_data, desired_shape=img_shape)
    #loaded_img_data = np.load('data/img_data{}.npy'.format(index))
    #loaded_numeric_data = np.load('data/numeric_data{}.npy'.format(index))




if __name__ == "__main__":
    main()

