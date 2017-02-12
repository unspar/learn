
import tensorflow as tf
import numpy as np
import unittest
import random

from seq_encoder import rnn_encode
from seq_decoder import rnn_decode
from tensorflow.python.ops import rnn_cell

def _parameterized_tester( inps, rank,batches, scope):
  '''
  called below to simplify each parameterized case
  '''

  for i in range(len(inps)):
    with tf.variable_scope('simple_test_'+str(i) + scope):
      int_target = inps[i]
      
      raw_data = tf.ones([batches, 5, rank], dtype=tf.float32)
      #this takes an array and converts into a list of 
      #seq_length elements of dimension: batch_size, rnn_size
      inputs = [tf.squeeze(i, [1]) for i in tf.split(1,5, raw_data)]
      target = tf.constant(int_target, dtype=tf.float32)
      encoded, state = rnn_encode(batches, rank,inputs)

      loss = tf.reduce_mean(tf.square(target - encoded))
      optimizer = tf.train.GradientDescentOptimizer(0.5)
      train = optimizer.minimize(loss)
      init = tf.global_variables_initializer()
      with tf.Session() as sess:
        init.run()
        for n in range(100):
          _ , out = sess.run([train, encoded])
      assert( abs(np.sum(out- int_target)) <= 0.05)

'''
class TestSequenceDecoder(unittest.TestCase):

  def test_simple_decode(self):
    
    cases = [
        [random.random() for i in range(5)],
        [5,4,3,2,1],
        [1,1,1,1,1],
        [1,10, 100, 1000, 10000]
    ]

    seq_len = 5
    for i in range(len(cases)):
      with tf.variable_scope('simple_test'+'dec'):

        rand = tf.random_uniform([1,5], 0,1)
        targets = tf.split(1,5,rand)
        seed = tf.split(1,5,rand)
        cell = rnn_cell.BasicRNNCell(1)  


        #seq_length elements of dimension: batch_size, rnn_size
        encodeds, state = rnn_decode(targets,cell.zero_state(1, tf.float32), cell)
        output = tf.concat(1, encodeds)
        tmp = rand
        tttt  = tf.square(tmp - output)
        loss = tf.reduce_sum(tf.square(tmp - output ))
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)
  
        with tf.Session() as sess:
          tf.initialize_all_variables().run()
          for n in range(100):
            _ , out, t, l, tt = sess.run([train, output, tmp, loss, tttt])
            print (l, out, t, tt)
            print( '===================')
        print (out)
        assert( abs(np.sum(out- cases[i])) <= 1)

'''

class TestSeqEncoder(unittest.TestCase):



  def test_simple_rnn_encode(self):
    #runs some very simple tests to check whether the model can match
    #a constant output
    
    _first_cases = [[0], [-1], [1], [100], [-100],[ 0.1], [-0.1]]

    _parameterized_tester(_first_cases, 1,1, 'simple')

  def test_two_dim_rnn_encode(self):
    #this performs tests idential to test_simple, but with two dimensions
    two_d_cases = [
      [0,0],
      [1,0], 
      [0,1],
      [100, -100], 
      [-100, 100]
    ]
    _parameterized_tester(two_d_cases, 2,1, '2d')


  def test_two_dim_batch_encode(self):
    '''
    this method checks to make sure that the batches work properly
    '''
    
    batch_cases = [
      [0,0],
      [1,0], 
      [0,1],
      [100, -100], 
      [-100, 100]
    ] 
    _parameterized_tester(batch_cases, 2,2,'2db')


unittest.main()

  






