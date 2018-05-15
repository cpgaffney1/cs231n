import os
from PIL import Image
import numpy as np
import util
import sys

def process_data_batch_and_write(batchnum, filenames, text_data, numeric_data):
    i = 0
    img_arrays = []
    x_shapes = []
    y_shapes = []
    zpid_list = {}
    for file in filenames:
        if i % 100 == 0:
            sys.stdout.write('=')
            sys.stdout.flush()
        try:
            img = Image.open('imgs/' + file)
        except OSError:
            print('file unreadable')
            continue
        data = np.array(img)
        if data.shape != (300, 400, 3):  # skip if improper shape. most are 300 x 300
            continue
        zpid = file[:-4]
        if zpid in text_data.keys() and zpid in numeric_data.keys():
            zpid_list[zpid] = i
        else:
            continue
        x_shapes.append(data.shape[0])
        y_shapes.append(data.shape[1])
        img_arrays.append(data)
        i += 1

    sys.stdout.write('>\n')
    sys.stdout.flush()

    N = len(set(text_data.keys()) & set(numeric_data.keys()) & set(zpid_list.keys()))
    print('N is {}'.format(N))
    assert(N == len(img_arrays))

    ordered_descriptions = [''] * N
    ordered_addresses = [''] * N
    ordered_numeric_data = np.zeros((N, 4))
    for zpid in zpid_list.keys():
        index = zpid_list[zpid]
        ordered_descriptions[index] = text_data[zpid][0]
        ordered_addresses[index] = text_data[zpid][1]
        ordered_numeric_data[index] = numeric_data[zpid]

    imgs = np.zeros((N, 300, 400, 3))
    for i in range(N):
        imgs[i] = img_arrays[i]

    mean_img = np.mean(imgs, axis=0)
    np.save('data/img_data{}.npy'.format(batchnum), imgs)
    np.save('data/numeric_data{}.npy'.format(batchnum), ordered_numeric_data)

    with open('data/descriptions{}.txt'.format(batchnum), 'w') as of:
        for y in ordered_descriptions:
            of.write('{}\n'.format(repr(y)))
    with open('data/addresses{}.txt'.format(batchnum), 'w') as of:
        for y in ordered_addresses:
            of.write('{}\n'.format(repr(y[1:-1])))

def process_data_batch(filenames, text_data, numeric_data, desired_shape=(299,299,3)):
    i = 0
    img_arrays = []
    x_shapes = []
    y_shapes = []
    zpid_list = {}
    print('Processing data batch')
    count = 0
    sys.stdout.write('<')
    sys.stdout.flush()
    for file in filenames:
        if count % 100 == 0:
            sys.stdout.write('=')
            sys.stdout.flush()
        count += 1
        try:
            img = Image.open('imgs/' + file)
        except OSError:
            print('file unreadable')
            continue
        data = np.array(img)
        if data.shape != (300, 400, 3):
            # skip if improper shape. most are 300 x 400
            continue
        zpid = file[:-4]
        if zpid not in text_data.keys() or zpid not in numeric_data.keys():
            continue
        x_shapes.append(data.shape[0])
        y_shapes.append(data.shape[1])
        assert(zpid not in zpid_list.keys())
        zpid_list[zpid] = i
        assert(i == len(img_arrays))
        img_arrays.append(data)
        i += 1

    assert(len(img_arrays) == len(zpid_list.keys()))
    sys.stdout.write('>\n')
    sys.stdout.flush()

    N = len(zpid_list)
    print('N is {}'.format(N))

    ordered_descriptions = [''] * N
    ordered_addresses = [''] * N
    ordered_numeric_data = np.zeros((N, 4))
    for zpid in zpid_list.keys():
        index = zpid_list[zpid]
        ordered_descriptions[index] = text_data[zpid][0]
        ordered_addresses[index] = text_data[zpid][1]
        ordered_numeric_data[index] = numeric_data[zpid]
    imgs = np.zeros((N, desired_shape[0], desired_shape[1], desired_shape[2]), dtype=np.uint8)
    for i in range(N):
        returned_image = util.crop(img_arrays[i], shape=(desired_shape[0], desired_shape[1]))
        imgs[i] = returned_image
        '''img = Image.fromarray(returned_image, 'RGB')
        img.show()
        print(returned_image)
        print('\n\n\n')
        print(imgs[i])
        img = Image.fromarray(imgs[i], 'RGB')
        img.show()
        exit()'''

    return imgs, ordered_numeric_data, ordered_descriptions, ordered_addresses


def load_tabular_data():
    numeric_data = {}
    text_data = {}
    with open('tabular_data/scraped_data.csv', encoding='utf8', errors='replace') as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            sp = line.split(';,.')
            zpid, zip, price, beds, baths, descr, address = sp
            numeric_data[zpid] = (zip, beds, baths, price)
            text_data[zpid] = (descr, address)
    return numeric_data, text_data

def main():
    numeric_data = {}
    text_data = {}
    with open('tabular_data/scraped_data.csv') as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            sp = line.split(';,.')
            zpid, zip, price, beds, baths, descr, address = sp
            numeric_data[zpid] = (zip, beds, baths, price)
            text_data[zpid] = (descr, address)

    index = 0
    batch_size = 1000
    files = os.listdir('imgs/')
    while len(files) != 0:
        process_data_batch_and_write(index, files[:batch_size], text_data, numeric_data)
        files = files[batch_size:]
        index += 1