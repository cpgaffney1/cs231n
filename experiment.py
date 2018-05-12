import numpy as np
from threading import Thread
from keras.models import Sequential
from sklearn import linear_model as sklm
import util
from model import build_model, Config
import os
import preprocessing

num_data_files = 50
n_epochs = 10

def main():
    print()
    ## run param search and other stuff
    #x_train, y_train, x_dev, y_dev, x_test, y_test = util.load_for_lin_reg()
    #reg = linear_regression(x_train, y_train, x_dev, y_dev, x_test, y_test)

    config = Config()
    model = build_model(config)
    train_model(model, config)


def linear_regression(x_train, y_train, x_dev, y_dev, x_test, y_test):
    reg = sklm.LinearRegression()
    reg.fit(x_train, y_train)
    return reg

def train_model(model, config):
    global loaded_img_data
    global loaded_numeric_data

    img_files = os.listdir('imgs/')

    #load initial data
    load_data_batch(img_files)
    img_data = loaded_img_data.copy()
    numeric_data = loaded_numeric_data.copy()

    data_indices = np.asarray(list(range(num_data_files)))
    np.random.shuffle(data_indices)
    #training loop
    for epoch in range(n_epochs):
        for index in data_indices:
            print('Fitting, epoch: {}'.format(epoch))
            #start loading data
            #data_thread = Thread(target=load_data_batch, args=(img_files,))
            #data_thread.start()

            # fit model on data batch
            model.fit([numeric_data[:200, 1:3], img_data[:200, :, :, :]],
                      util.buckets(numeric_data[:200, 3], num=config.n_classes),
                      batch_size=config.batch_size, validation_split=0.1)

            #retrieve new data
            #data_thread.join()
            #img_data = loaded_img_data.copy()
            #numeric_data = loaded_numeric_data.copy()


loaded_img_data = None
loaded_numeric_data = None
def load_data_batch(img_files, batch_size=1000):
    global loaded_img_data
    global loaded_numeric_data
    numeric_data, text_data = preprocessing.load_tabular_data()
    imgs, numeric_data, descriptions, addresses= \
        preprocessing.process_data_batch(np.random.choice(img_files, size=batch_size), text_data, numeric_data)
    loaded_img_data = imgs
    loaded_numeric_data = numeric_data
    #loaded_img_data = np.load('data/img_data{}.npy'.format(index))
    #loaded_numeric_data = np.load('data/numeric_data{}.npy'.format(index))




if __name__ == "__main__":
    main()

