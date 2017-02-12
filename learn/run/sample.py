from __future__ import print_function

from model import Model
import tensorflow as tf
import char_dataset as cd
import random 
dataset = cd.Dataset('./data/shakespeare.txt')


sl = 25

def main():
  
  model = Model(1,sl , dataset.distinct_characters ,train=False) #true to sample

  sess = tf.Session()
  tf.initialize_all_variables().run(session = sess)
  saver = tf.train.Saver(tf.all_variables())


  #grab sl characters to encode as a sample
  dataset.place = random.randint(0,100000)
  
  for i in range(10):
    chars = dataset.char_to_num(dataset.readn(sl))
    
    sampled = model.sample(sess, saver,  chars)
    
    sampled_str = ''.join(dataset.num_to_char(sampled))
    print("=======Sampled sentence=======")
    print(''.join(dataset.num_to_char(chars)))

    print("=======Encoded sentence=======")
    print(sampled_str)

if __name__ == '__main__':
    main()
