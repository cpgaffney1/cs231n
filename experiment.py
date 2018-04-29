import numpy as np
from threading import Thread
from keras.models import Sequential

num_data_files = 50
n_epochs = 10

def main():
    print()
    ## run param search and other stuff


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


if __name__ == "__main__":
    main()