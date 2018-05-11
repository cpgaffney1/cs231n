import numpy as np


def partition(x, num):
    bins = np.linspace(0,np.max(x), num=num)
    print(bins)
    y = np.digitize(x,bins, right=True)
    return y

