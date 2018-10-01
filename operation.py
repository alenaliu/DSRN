import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
import skimage.io as skio
import re
import skimage.transform
import skimage.color as color
import random


def shuffle(bbox,data,img):
    nums=bbox.shape[0]
    ron=range(nums)
    random.shuffle(ron)
    bbox=bbox[ron,:]
    data=data[ron,:]
    img=img[ron,:,:,:]
    return bbox,data,img
