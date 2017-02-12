import numpy as np


class ModelFunctions():
  '''
  functions used by the Model class.
  Mostly static methods
  '''
  
  def weighted_pick( self,weights):
    '''
    Takes a vector of weights
    returns an index into weights randomly seleted
    with the value of weight in weights determining the probability
    '''
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1)*s)))


  @staticmethod
  def tf_weighted_pick(weights, scope = None):
    '''
    Takes a vector of weights
    returns an index into weights randomly seleted
    with the value of weight in weights determining the probability
    '''
    with variable_scope.variable_scope(scope or "weighted_sample"):
      sum =  tf.redice_sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1)*s)))


