'''
Import
'''


import numpy as np

import scipy.misc
import img_dataset as ig
import tensorflow as tf
import autoencoder as md


ds = md.Autoencoder(28*28, sample=True)

ds.sample()




