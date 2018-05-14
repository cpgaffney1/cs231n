import numpy as np
import util
from PIL import Image


def partition(x, num):
    bins = np.linspace(0,np.max(x), num=num)
    print(bins)
    y = np.digitize(x,bins, right=True)
    return y

if __name__ == "__main__":
    #img = np.random.random((10,10,3))
    img = Image.open('C:/Users/Juan/Documents/17-18 _Stanford/Port 2A/ilha.jpg')
    img = np.array(img)
    new_img = util.crop(img, (300,300), True)
    print(new_img.shape)
    img = Image.fromarray(new_img, 'RGB')
    img.show()
    print(new_img)
    print('\n\n\n')
    #print(imgs[i])
    #img = Image.fromarray(imgs[i], 'RGB')
    #img.show()
    exit()