import numpy as np
import os
import shutil

if os.path.isdir('test_imgs/'):
    shutil.rmtree('test_imgs/')
os.mkdir('test_imgs/')

img_files = os.listdir('imgs/')
test_files = np.random.choice(img_files, size=int(0.1 * len(img_files)))

for file in test_files:
    shutil.move('imgs/' + file, 'test_imgs/')