import numpy as np
from threading import Thread
from keras.models import Sequential
from sklearn import linear_model as sklm

num_data_files = 50
n_epochs = 10

def main():
    print()
    ## run param search and other stuff
    linear_regression()

def linear_regression():
    with open('C:/Users/cpgaf/PycharmProjects/zillow_scraper/scraped_data.csv') as f:
        lines = f.readlines()
        lines = lines[1:]
        X = np.zeros((len(lines), 2))
        y = np.zeros(len(lines))
        for i, line in enumerate(lines):
            sp = line.split(';,.')
            zpid, zip, price, beds, baths, descr, address = sp
            X[i][0] = int(beds)
            X[i][1] = int(baths)
            y[i] = float(price)

    X, y = shuffle_in_parallel(X, y)
    x_train, y_train, x_dev, y_dev, x_test, y_test = split_data(X, y)

    reg = sklm.LogisticRegression()
    reg.fit(x_train, y_train)





    print()
    print('Model weights')

def build_train_model(config):
    global loaded_img_data
    global loaded_numeric_data

    #load initial data
    load_data_batch(0)
    img_data = loaded_img_data.copy()
    numeric_data = loaded_numeric_data.copy()

    # build model
    print()

    data_indices = np.random.shuffle(list(range(num_data_files)))
    #training loop
    for _ in range(n_epochs):
        for index in data_indices:
            #start loading data
            data_thread = Thread(target=load_data_batch, args=(index,))
            data_thread.start()

            # fit model on data batch


            #retrieve new data
            data_thread.join()
            img_data = loaded_img_data.copy()
            numeric_data = loaded_numeric_data.copy()

def build_model(config):
    model = Sequential()

loaded_img_data = None
loaded_numeric_data = None
def load_data_batch(index):
    global loaded_img_data
    global loaded_numeric_data
    loaded_img_data = np.load('data/img_data{}.npy'.format(index))
    loaded_numeric_data = np.load('data/numeric_data{}.npy'.format(index))


def shuffle_in_parallel(arr1, arr2):
    assert(len(arr1) == len(arr2))
    indices = np.arange(len(arr1))
    np.random.shuffle(indices)
    return arr1[indices], arr2[indices]



#splits train, dev, test
def split_data(x, y):
    #train, dev, test
    split_portions = [0.7, 0.1, 0.1]
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


if __name__ == "__main__":
    main()