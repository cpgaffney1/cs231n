import os
from PIL import Image
import numpy as np

i = 0

img_arrays = []
x_shapes = []
y_shapes = []
data_path = 'C:/Users/cpgaf/PycharmProjects/zillow_scraper'
for file in os.listdir(data_path + '/imgs'):
    if i % 100 == 0:
        print('iteration {}'.format(i))
    try:
        img = Image.open(data_path + '/imgs/' + file)
    except OSError:
        print('file unreadable')
        continue
    data = np.array(img)
    '''if data.shape[0] != 300:
        back = Image.fromarray(data, 'RGB')
        back.show()
        exit()'''
    if data.shape != (300, 300, 3): #skip if improper shape. most are 300 x 300
        continue
    x_shapes.append(data.shape[0])
    y_shapes.append(data.shape[0])
    img_arrays.append(data)
    i += 1

imgs = np.zeros((len(img_arrays), 300, 300, 3))
for i in range(len(img_arrays)):
    imgs[i] = img_arrays[i]

mean_img = np.mean(imgs, axis=0)

with open('xout.txt', 'w') as of:
    for x in x_shapes:
        of.write('{}\n'.format(x))
with open('yout.txt', 'w') as of:
    for y in y_shapes:
        of.write('{}\n'.format(y))




