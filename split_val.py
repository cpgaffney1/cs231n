import numpy as np
import os
import shutil

if os.path.isdir('val_imgs/'):
    shutil.rmtree('val_imgs/')
os.mkdir('val_imgs/')

img_files = os.listdir('imgs/')
test_files = np.random.choice(img_files, size=int(0.1 * len(img_files)), replace=False)

for file in test_files:
    shutil.move('imgs/' + file, 'val_imgs/')