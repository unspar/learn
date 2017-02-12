'''
This contains functions to work out embeddings


'''

import numpy as np
import tensorflow as tf

def sequence_lookup(input_tensor, output_size, scope=None):
  '''
  Takes an input tensor and scope
  returns the tensor graph with a lookup performed
  Args:
    input_tensor - a tensor
  '''

  with tf.variable_
  

  t = np.cumsum(weights)
  s = np.sum(weights)
  return(int(np.searchsorted(t, np.random.rand(1)*s)))


def tf_weighted_pick(weights, scope = None):
  '''
  Takes a vector of weights
  returns an index into weights randomly seleted
  with the value of weight in weights determining the probability
  '''
  with variable_scope.variable_scope(scope or "weighted_sample"):
    sum =  tf.redice_sum(weights)
  return(int(np.searchsorted(t, np.random.rand(1)*s)))


