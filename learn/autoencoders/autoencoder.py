import tensorflow as tf
from tensorflow.python.ops import variable_scope


import numpy as np
import scipy.misc as scmi
import scipy.special as scisp 

import img_dataset as img


import os
import time
import random


DATATYPE = tf.float32

def var_init(fan_in, fan_out, constant=1):
  """ Xavier initialization of network weights"""
  # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
  low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
  high = constant*np.sqrt(6.0/(fan_in + fan_out))
  return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

#TODO- throw this in its own module for simple nn layers
def two_layer_ann(inputs, input_size, hidden_size, encoded_size, scope=None):

    
  #default settings
  activation_function =tf.nn.softplus

  with variable_scope.variable_scope(scope or "autoencoder_hidden_layer"):
    
    h1  =  tf.Variable(var_init(input_size, hidden_size))
    h1b= tf.Variable(tf.zeros([hidden_size], dtype=DATATYPE))
    h2 = tf.Variable(var_init(hidden_size, hidden_size))
    h2b = tf.Variable(tf.zeros([hidden_size], dtype=DATATYPE))


    eh1 = activation_function( tf.add(tf.matmul(inputs,h1), h1b))
    eh2 = activation_function( tf.add(tf.matmul(eh1, h2), h2b))

  return eh2


def linear_trans(tf_input, hidden_size, encoded_size, scope= None):

  
  with variable_scope.variable_scope(scope or "linear_transformation"):
    weights  = tf.Variable(var_init(hidden_size, encoded_size ))
    bias = tf.Variable(tf.zeros([encoded_size],dtype=DATATYPE))
    
  return tf.add(tf.matmul(tf_input, weights), bias)

class Autoencoder():
  '''
  This class contains an autoencoder intended to learn
  an encoding and a decoding of presented data

  It is intended as a method for producing a reversable embedding
  to help with the processing of sequence data
  '''
  save_dir = "save"
  
  num_epochs = 10
  epoch_size =100

  hidden_size = 500
  encoded_size = 20

  num_layers = 2
  
  batch_size = 500
  
  grad_clip = 5

  learning_rate = 0.001
  decay_rate = 0.97



  def __init__(s,dimm,  sample = False):
    '''
    Builds the tensorflow autoencoder
    '''
    s.dataset = img.ImgDataset("train-images-idx3-ubyte.gz")
    tf.set_random_seed(0)

    if sample == True:
      s.bs = 1
    else:
      s.bs = s.batch_size 
    #alais to save linespace
    f = tf.float32
    #batch size images, and the serialized input data

    s.dimm = dimm
    s.inp = tf.placeholder(tf.float32, shape=(None, s.dimm))
    ac_fun = tf.nn.softplus
    
    #From hidden layer generate mu and sigma using ac_fun (? why this? to prevent negative values?)
    s.eh2 = two_layer_ann(s.inp, s.dimm, s.hidden_size, s.encoded_size)
    #mu is mean of gaussian
    #sigma is log standard deviation of the gaussian (? why lg?)
    #ac_fun taken from https://jmetzen.github.io/2015-11-27/vae.html
    s.ln_sigma_sq = linear_trans(s.eh2, s.hidden_size, s.encoded_size, scope='ln_sigma_sq')
    s.mu = linear_trans(s.eh2, s.hidden_size, s.encoded_size, scope='mu')
    
    
    #Now we sample from our latent space (randomness comes from eps
    # z = mu + sigma*epsilon
    # (?why exp and then sqrt is that to turn it back into sigma?)
    #NOTE - tf.mul vs tf.matmul
    #this randomness helps to make the prob model work 

    s.eps = tf.random_normal((s.bs, s.encoded_size ), 0, 1, dtype=f)
    s.encoded = tf.add(s.mu, tf.mul(tf.sqrt(tf.exp(s.ln_sigma_sq)), s.eps))
    
    #######################################3
    #   DECODING LAYER
    #Decoding the encoded layer
    s.dh2 = two_layer_ann(s.encoded, s.encoded_size, s.hidden_size, s.dimm)

    #output image
    #sigmoid here?
    #adding a sigmoid here makes the input and output of identical forms. is that valuable?
    out= tf.Variable(var_init(s.hidden_size, s.dimm))
    outb= tf.Variable(tf.zeros([s.dimm],dtype=f))
    s.output =tf.nn.sigmoid((tf.add(tf.matmul(s.dh2, out), outb)))
    
    #TODO- Undersand this
    # The loss is composed of two terms:
    # 1.) The reconstruction loss (the negative log probability
    #     of the input under the reconstructed Bernoulli distribution 
    #     induced by the decoder in the data space).
    #     This can be interpreted as the number of "nats" required
    #     for reconstructing the input when the activation in latent
    #     is given.
    # Adding 1e-10 to avoid evaluatio of log(0.0)
    #s.reconstr_loss = -tf.reduce_sum((s.inp * tf.log(1e-10 + s.output)) + ((1-s.inp) * tf.log(1e-10 + 1 - s.output)), 1)
    s.reconstr_loss = -tf.reduce_sum(s.inp * tf.log(1e-10 + s.output) + (1-s.inp) * tf.log(1e-10 + 1 - s.output),1)
    # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
   ##    between the distribution in latent space induced by the encoder on 
    #     the data and some prior. This acts as a kind of regularizer.
    #     This can be interpreted as the number of "nats" required
    #     for transmitting the the latent space distribution given
    #     the prior.

    #s.latent_loss = -0.5 * tf.reduce_sum(1 + s.log_sig - tf.square(s.mu) - tf.exp(s.log_sig), 1)
    #s.kl_divergence = -0.5 * tf.reduce_sum(1+ s.log_sig_sq - tf.square(s.mu) - tf.exp(s.log_sig_sq), 1) 
    s.latent_loss = -0.5 * tf.reduce_sum(1 + s.ln_sigma_sq - tf.square(s.mu)- tf.exp(s.ln_sigma_sq), 1)

    #s.cost = 0.5* tf.reduce_mean(tf.pow(s.inp - s.output, 2)) 
    #s.cost = tf.reduce_mean(s.kl_divergence + s.reconstr_loss)   # average over batch
    s.cost = tf.reduce_mean(s.reconstr_loss + s.latent_loss)   # average over batch   

 
    #learning rate
    #s.lr = tf.Variable(0.0, trainable=False)
    #tvars = tf.trainable_variables()
    ##grads, _ = tf.clip_by_global_norm(tf.gradients(s.cost, tvars), s.grad_clip)
    #note, optomizer.minimize also could work here
    s.optimizer = tf.train.AdamOptimizer(learning_rate=s.learning_rate).minimize(s.cost)
    #s.train_op = s.optimizer.apply_gradients(zip(grads, tvars)) 
    #s.optimizer = tf.train.AdamOptimizer(learning_rate=s.learning_rate).minimize(s.cost)
 
  def train(s):

    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      saver = tf.train.Saver(tf.all_variables())
      #TODO- work out hwo to restore a model from a save
      # restore model (if applicable)
      # saver.restore(sess, ckpt.model_checkpoint_path)
      for e in range(s.num_epochs):
        #sess.run(tf.assign(s.lr, s.learning_rate* s.decay_rate*e))
        s.dataset.reset()    
         
        save_dir = os.path.join(s.save_dir, 'model.ckpt')
        saver.save(sess,save_dir, global_step = e) 
        print("saving to " +s.save_dir)
        for n in range(s.epoch_size):
          start = time.time()  
          
          #this clip is important to normalize the data before processing
          feed = {s.inp: np.clip(s.dataset.readn(s.bs), 0, 1) }
          _, train_loss, hidden = sess.run([s.optimizer, s.cost, s.reconstr_loss], feed)
          end = time.time()
          print("batch: {}, epoch: {}, train_loss = {:.3f}, time/batch = {:.3f}" \
                   .format(e * s.epoch_size + n,
                         e, train_loss, end - start))


  def sample(s):
    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      saver = tf.train.Saver(tf.all_variables())
      
      ckpt = tf.train.get_checkpoint_state(s.save_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print("loaded model from : " + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path) 

      if s.bs != 1 : 
        raise ValueError("can only sample from models with a batch size and sequence length of one" )
      
      burn = s.dataset.readn(random.randint(1, 100) )
      img = np.clip(s.dataset.readn(1),0,1)
      feed = {s.inp:img}
      inp, output = sess.run([s.inp, s.output], feed)
      i = np.multiply(255, np.reshape(inp,(28,28))).astype('uint8')
      o = np.multiply(255, np.reshape(output,(28,28))).astype('uint8')
      
      scmi.imsave('output.jpg', o)
      scmi.imsave('input.jpg', i)



 
