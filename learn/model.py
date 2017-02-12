'''
This is an RNN autoencoder.

It learns an encoding of a string into a feature space
by encoding and decoding it repeatedly.

'''

import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope

from rnn_utils import unroll

from utility_functions import ModelFunctions as mf


class Model():

  #TODO - move this into JSON?
  rnn_size = 50
  num_layers = 2
  
  grad_clip = 5
  learning_rate = 0.005
  decay_rate = 0.97

  save_dir = "save"

  def __init__(s, batch_size, seq_length, vocab_size, train = True):
    s.bs = bs = batch_size
    s.sl = sl = seq_length
    s.vocab_size = vocab_size
    f=tf.float32 
    with tf.variable_scope('text_autoencoder') as model_scope:
      #TODO- Migrate to actual LTSM Cell. 

      #input 
      s.input_raw = tf.placeholder(tf.int32, [bs, sl])
      s.targets = tf.placeholder(tf.int32, [bs, sl])

      with tf.variable_scope('encoder'):
        enc_cell =  rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(s.rnn_size)] * s.num_layers)
        #embedding lookup is faster than rnn
        embedding = tf.get_variable('embedding', [s.vocab_size, s.rnn_size]) 

        embedded_inputs = tf.split(1, sl, tf.nn.embedding_lookup(embedding, s.input_raw))
        embedded_inputs = [tf.squeeze(input_, [1]) for input_ in embedded_inputs]

        _, encoded = rnn.rnn(enc_cell, embedded_inputs, initial_state=enc_cell.zero_state(bs, f))

      with tf.variable_scope('decoder'):
        dec_cell =  rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(s.rnn_size)] * s.num_layers)
       
        #this is used to translate the hidden space into the
        #actual space we know of
        dec_w = tf.get_variable("softmax_w", [s.rnn_size, s.vocab_size])
        dec_b = tf.get_variable("softmax_b", [s.vocab_size])

        #this is used as the first input
        spacer = tf.zeros([s.bs, s.rnn_size], f)

        #in the case where we're training, the expected decoder inputs
        #are identical to the inputs
        #if we're sampling, we need to sample and decode (sadness)
        if train:
          sample_outputs = [spacer]+ embedded_inputs[0:-1]
          decoded, _ = rnn.rnn(dec_cell, sample_outputs, initial_state=encoded)
          output = tf.reshape(tf.concat(1,decoded), [-1, s.rnn_size])
          logits = tf.matmul(output, dec_w) + dec_b
          s.probs = tf.nn.softmax(logits)

          #TODO- replace this magic
          loss = seq2seq.sequence_loss_by_example([logits],
                  [tf.reshape(s.targets, [-1])],
                  [tf.ones([bs * sl])],
                  s.vocab_size)
          s.cost = tf.reduce_sum(loss) / bs/ sl
          s.lr = tf.Variable(0.0, trainable=False)
          tvars = tf.trainable_variables()
          grads, _ = tf.clip_by_global_norm(tf.gradients(s.cost, tvars),
                  s.grad_clip)
          optimizer = tf.train.AdamOptimizer(s.lr)
          s.train_op = optimizer.apply_gradients(zip(grads, tvars))
        else:
          #this block handles the sampling loop
          probs = []
          
          keys = tf.constant(range(s.sl)) #keys are used used in random choice below
          #initial state is the encoded value w/spacer
          re_encoded = spacer
          state = encoded
          for i in range(s.sl):
            if i > 0:
              variable_scope.get_variable_scope().reuse_variables()
            raw_decoded, state = dec_cell(re_encoded,state)
            #combines logits and probs from first if 
            odds  = tf.nn.softmax(tf.matmul(raw_decoded, dec_w) + dec_b)
            probs.append(odds)
            index = tf.multinomial(tf.log(odds), num_samples=1)
            re_encoded = tf.expand_dims(tf.nn.embedding_lookup(embedding, tf.squeeze(index) ), 0)
          s.probs = tf.concat(0, probs)


  
  def sample(s,sess, saver,inp):
    '''
     encodes the string in chars
     returns the decoded value
     Args
       sess - tensorflow session to use
       saver - saver utility to use
       state - model state to use
       inp- characters to encode
    
    '''
    #restore latest save from the save directory
    ckpt = tf.train.get_checkpoint_state(s.save_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print("loaded model from : " + ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
    
    if s.bs != 1: 
      raise ValueError("can only sample from models with a batch size and sequence length of one" )
    inp =  np.reshape(inp, (s.bs, s.sl))
    feed = {s.input_data: inp}
    #oxutput is s.sl X distinct_chars
    output = sess.run([s.probs], feed)
    #outputs is a 1 x sl x dimm
    outputs = np.split(output[0], s.sl)
    ret = []
    for n in range(s.sl):
      char = mf.weighted_pick(outputs[n])
      
      ret.append(int(np.squeeze(char)))
    return  ret



