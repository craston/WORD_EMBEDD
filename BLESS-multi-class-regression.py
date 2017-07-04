
""" Logistic regression for multiclass classification
Class labels
Hypernyms   = 0
Co-siblings = 1
Random      = 2

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import gensim
import numpy as np

FLAGS = None


# function which generates WORD vectors and returns training and test feature vectors
def word_embeddding():
  fname = "datasets/BLESS/BLESS_hyper-new4.txt"

  model = gensim.models.KeyedVectors.load_word2vec_format('~/MoSIG/2016_data/S2/Summer_internship/Models/GoogleNews-vectors-negative300.bin', binary=True)  

  with open(fname) as f:
      hyper = f.readlines()
  # you may also want to remove whitespace characters like `\n` at the end of each line
  hyper = [x.strip('\n') for x in hyper] 
  hyper = [x.split('\t') for x in hyper] 
  hyper0 = [x[0].split('-n')[0] for x in hyper] 
  hyper1 = [x[2].split('-n')[0] for x in hyper] 
 
  v0 = [model[x] for x  in hyper0]        # Generate vector of word1 of pair  (dimension = 300)
  v1 = [model[x] for x in hyper1]         # Generate vector of word2 of pair  (dimension = 300)

  #print("HYPERNYMS")
  #print(np.array(v0).shape)               # Converting to float32 numpy array
  v0 = np.array(v0, dtype = np.float32)         
  v1 = np.array(v1, dtype = np.float32)

  v_hyp = np.concatenate((v0,v1), axis=1) # Generating feature vector for word pair by concatenating the vectors (dimension = 600)
  labels_hyp = np.zeros(v_hyp.shape[0], dtype=np.int)   # Class label for hypernym = 0
  #print (labels_hyp.size)
  #print (v_hyp.shape)

  fname = "datasets/BLESS/BLESS_coord-new6.txt"

  with open(fname) as f:
      hyper = f.readlines()
  # you may also want to remove whitespace characters like `\n` at the end of each line
  hyper = [x.strip('\n') for x in hyper] 
  hyper = [x.split('\t') for x in hyper] 
  hyper0 = [x[0].split('-n')[0] for x in hyper] 
  hyper1 = [x[2].split('-n')[0] for x in hyper] 
  hyper2 = [x[1] for x in hyper] 

  v0 = [model[x] for x  in hyper0]        # Generate vector of word1 of pair  (dimension = 300)
  v1 = [model[x] for x in hyper1]         # Generate vector of word2 of pair  (dimension = 300)

  #print("CO SIBLINGS")
  #print(np.array(v0).shape)
  v0 = np.array(v0, dtype = np.float32)
  v1 = np.array(v1, dtype = np.float32)

  v_coord = np.concatenate((v0,v1), axis=1) # Generating feature vector for word pair by concatenating the vectors (dimension = 600)
  labels_coord = np.empty(v_coord.shape[0], dtype=np.int)
  labels_coord.fill(1)                    # Class label for co-sibling = 1
  #print (labels_coord.size)
  #print (v_coord.shape)


  fname = "datasets/BLESS/BLESS_random-new5.txt"

  with open(fname) as f:
      hyper = f.readlines()
  # you may also want to remove whitespace characters like `\n` at the end of each line
  hyper = [x.strip('\n') for x in hyper] 
  hyper = [x.split('\t') for x in hyper] 
  hyper0 = [x[0].split('-n')[0] for x in hyper] 
  hyper1 = [x[2].split('-n')[0] for x in hyper] 
  hyper2 = [x[1] for x in hyper] 

  v0 = [model[x] for x  in hyper0]
  v1 = [model[x] for x in hyper1]

  #print("RANDOM")
  #print(np.array(v0).shape)
  v0 = np.array(v0, dtype = np.float32)
  v1 = np.array(v1, dtype = np.float32)

  v_rand = np.concatenate((v0,v1), axis=1)
  labels_rand = np.empty(v_rand.shape[0], dtype=np.int32)
  labels_rand.fill(2)                       # Class label for random = 2
  #print (labels_rand.size)
  #print (v_rand.shape)

  v_final = np.concatenate((v_hyp, v_rand, v_coord), axis=0)      # Merging all vectors
  labels_final = np.concatenate((labels_hyp, labels_rand, labels_coord), axis=0)
  labels_final = np.expand_dims(labels_final, axis=1)

  #print("FINAL")
  #print(v_final.shape)
  #print(labels_final.shape) 

  BIG = np.concatenate((v_final, labels_final), axis=1)
  #print(BIG.shape)
  np.random.shuffle(BIG)                                          # Shuffling the dataset

  #print(BIG.shape)
  EMBEDD = BIG[:, 0:600]
  LAB = BIG[:,600]

  #print(LAB[1:50])
  LAB = np.int32(LAB)
  LAB = np.expand_dims(LAB, axis=1)
  

  #one-hot encoding for the labels
  LABELS = np.zeros((len(LAB), 3))
  LABELS[np.arange(len(LAB)), LAB[:,0]] = 1
  #print(LABELS.shape)
  #print(LABELS[1:50,:])
  return EMBEDD[:4094,:], LABELS[:4094,:], EMBEDD[4095:,:], LABELS[4095:,:]   # splitting training 4094 pairs, test 

def main(_):

  # Create the model
  x = tf.placeholder(tf.float32, [None, 600])
  W = tf.Variable(tf.zeros([600, 3]))
  b = tf.Variable(tf.zeros([3]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 3])

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  batch_xs, batch_ys, test_xs, test_ys = word_embeddding()  #Using regular gradient descent without batching i.e using the entire training set for each update
  
  # Train
  for _ in range(1000):
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model after each iteration
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys}))

if __name__ == '__main__':
  tf.app.run(main=main)
