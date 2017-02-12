from __future__ import print_function

from model import Model

import char_dataset as cd
import numpy as np
import tensorflow as tf
import os
dataset = cd.Dataset('./data/shakespeare.txt')


#batch size and sequence length
#these are hyperparameters?
bs = 50
sl= 50


def main():
    train()


def train():


  model = Model(bs, sl, dataset.distinct_characters)
  sess = tf.Session()
  tf.initialize_all_variables().run(session= sess)
  saver = tf.train.Saver(tf.all_variables())
  save_dir = os.path.join(model.save_dir, 'model.ckpt')

  
  #number of epochs to run
  for n in range(100):
    model.save(sess, saver, save_dir, n)
    print("epoch {}".format(n))
    sess.run(tf.assign(model.lr, model.learning_rate* model.decay_rate * n))
    #this resets the state to zero_state (why?)
    print('epoch: {}'.format(n))
    for nt in range(500): 
      chars = dataset.char_to_num(dataset.readn(sl *bs))
      inps= np.reshape(np.copy(chars), (bs,sl))
      exps = np.copy(inps)# np.reshape(np.copy(chars[1:(bs*sl)+1]),(bs, sl))
      state= model.train(sess, inps, exps)

  return


if __name__ == '__main__':
    main()
