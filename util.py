import numpy as np

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

def crop(image, shape=(299, 299, 3)):
    new_img = np.zeros((shape[0], shape[1], shape[2]))
    for i in range(len(image.shape)):
        assert(image.shape[i] >= shape[i])
    '''new_img[:,:,:] = image[(image.shape[0] - shape[0])//2 : (image.shape[0] - shape[0])//2 + shape[0],
                     (image.shape[1] - shape[1]) // 2: (image.shape[1] - shape[1]) // 2 + shape[1],
                     :]'''
    new_img[:,:,:] = image[:shape[0], :shape[1], :]
    return new_img

def buckets(x, num):
    bins = np.linspace(0,np.max(x), num=num)
    print(bins)
    y = np.digitize(x,bins, right=True)
    return y