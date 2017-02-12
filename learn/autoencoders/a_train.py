'''
Import
'''


import numpy as np

import scipy.misc
import img_dataset as ig
import tensorflow as tf
import autoencoder as md

print('beginning training')
minst = ig.ImgDataset("train-images-idx3-ubyte.gz")

ds = md.Autoencoder(28*28)

ds.train()


print('training complete')


