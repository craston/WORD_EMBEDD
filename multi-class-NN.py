
""" Neural network for multiclass classification
Class labels
Hypernyms   = 0
Co-siblings = 1
Random      = 2

USAGE 
python multi-class-NN.py --pretrained Models/GoogleNews-vectors-negative300.bin --dataset ROOT9 --split 0.1 --balance 0 --learning_rate 0.5

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

  model = gensim.models.KeyedVectors.load_word2vec_format('{0}'.format(args.pretrained), binary=True) 
  #========== HYPERNYM vector generation ======================#
  fname = "datasets/{0}/{0}_hyper-new.txt".format(args.dataset)

  with open(fname) as f:
    hyper = f.readlines()
  # you may also want to remove whitespace characters like `\n` at the end of each line
  hyper = [x.strip('\n') for x in hyper] 
  hyper = [x.split('\t') for x in hyper] 
  hyper0 = [x[0].split('-n')[0] for x in hyper] 
  hyper1 = [x[2].split('-n')[0] for x in hyper] 

  v_hyper0 = [model[x] for x  in hyper0]        # Generate vector of word1 of pair  (dimension = 300)
  v_hyper1 = [model[x] for x in hyper1]         # Generate vector of word2 of pair  (dimension = 300)

  # Converting to float32 numpy array
  v_hyper0 = np.array(v_hyper0, dtype = np.float32)         
  v_hyper1 = np.array(v_hyper1, dtype = np.float32)

  v_hyper = np.concatenate((v_hyper0, v_hyper1), axis=1) # Generating feature vector for word pair by concatenating the vectors (dimension = 600)
  labels_hyper = np.zeros(v_hyper.shape[0], dtype=np.int)   # Class label for hypernym = 0

  #========== CO-SIBLING vector generation ======================#
  fname = "datasets/{0}/{0}_coord-new.txt".format(args.dataset)

  with open(fname) as f:
    coord = f.readlines()
  # you may also want to remove whitespace characters like `\n` at the end of each line
  coord = [x.strip('\n') for x in coord] 
  coord = [x.split('\t') for x in coord] 
  coord0 = [x[0].split('-n')[0] for x in coord] 
  coord1 = [x[2].split('-n')[0] for x in coord] 

  v_coord0 = [model[x] for x  in coord0]        # Generate vector of word1 of pair  (dimension = 300)
  v_coord1 = [model[x] for x in coord1]         # Generate vector of word2 of pair  (dimension = 300)
  v_coord0 = np.array(v_coord0, dtype = np.float32)
  v_coord1 = np.array(v_coord1, dtype = np.float32)

  v_coord = np.concatenate((v_coord0,v_coord1), axis=1) # Generating feature vector for word pair by concatenating the vectors (dimension = 600)
  labels_coord = np.empty(v_coord.shape[0], dtype=np.int)
  labels_coord.fill(1)                    # Class label for co-sibling = 1


  #========== RANDOM vector generation ======================#
  fname = "datasets/{0}/{0}_random-new.txt".format(args.dataset)

  with open(fname) as f:
    rand = f.readlines()
  # you may also want to remove whitespace characters like `\n` at the end of each line
  rand = [x.strip('\n') for x in rand] 
  rand = [x.split('\t') for x in rand] 
  rand0 = [x[0].split('-n')[0] for x in rand] 
  rand1 = [x[2].split('-n')[0] for x in rand] 

  v_rand0 = [model[x] for x  in rand0]        # Generate vector of word1 of pair  (dimension = 300)
  v_rand1 = [model[x] for x in rand1]         # Generate vector of word2 of pair  (dimension = 300)
  v_rand0 = np.array(v_rand0, dtype = np.float32)
  v_rand1 = np.array(v_rand1, dtype = np.float32)

  v_rand = np.concatenate((v_rand0,v_rand1), axis=1) # Generating feature vector for word pair by concatenating the vectors (dimension = 600)
  labels_rand = np.empty(v_rand.shape[0], dtype=np.int)
  labels_rand.fill(2)                    # Class label for rand = 2

  #================splitting into training and test =================#
  
  #Balancing the pairs:
  if(args.balance == 1):
    least_number = min([len(labels_hyper), len(labels_coord), len(labels_rand)])
    #print("least number = {0}".format(least_number))

    v_hyper = v_hyper[:least_number,:]
    v_coord = v_coord[:least_number,:]
    v_rand  = v_rand[:least_number,:]
    labels_hyper = labels_hyper[:least_number]
    labels_coord = labels_coord[:least_number]
    labels_rand  = labels_rand[:least_number]
    #print("testing length = {0}".format(round(args.split*len(labels_hyper))))
  
  v_hyper_train      = v_hyper[:round(args.split*len(labels_hyper)),:]
  v_hyper_test       = v_hyper[round(args.split*len(labels_hyper))+1:,:]
  labels_hyper_train = labels_hyper[:round(args.split*len(labels_hyper))]
  labels_hyper_test  = labels_hyper[round(args.split*len(labels_hyper))+1:]

  v_coord_train = v_coord[:round(args.split*len(labels_coord)),:]
  v_coord_test  = v_coord[round(args.split*len(labels_coord))+1:,:]
  labels_coord_train  = labels_coord[:round(args.split*len(labels_coord))]
  labels_coord_test  = labels_coord[round(args.split*len(labels_coord))+1:]

  v_rand_train = v_rand[:round(args.split*len(labels_rand)),:]
  v_rand_test  = v_rand[round(args.split*len(labels_rand))+1:,:]
  labels_rand_train  = labels_rand[:round(args.split*len(labels_rand))]
  labels_rand_test  = labels_rand[round(args.split*len(labels_rand))+1:]

  #========== Merging all the data ===================================#
  v_train = np.concatenate((v_hyper_train, v_coord_train, v_rand_train), axis=0)      
  labels_train = np.concatenate((labels_hyper_train, labels_coord_train, labels_rand_train), axis=0)
  labels_train = np.expand_dims(labels_train, axis=1)

  v_test = np.concatenate((v_hyper_test, v_coord_test, v_rand_test), axis=0)     
  labels_test = np.concatenate((labels_hyper_test, labels_coord_test, labels_rand_test), axis=0)
  labels_test = np.expand_dims(labels_test, axis=1)

  # Shuffling the dataset
  BIG_train = np.concatenate((v_train, labels_train), axis=1)
  np.random.shuffle(BIG_train)                                          
  BIG_test = np.concatenate((v_test, labels_test), axis=1)
  np.random.shuffle(BIG_test)                                         

  EMBEDD_train = BIG_train[:, 0:600]
  LAB_train = np.expand_dims(np.int32(BIG_train[:,600]), axis=1)
  EMBEDD_test = BIG_test[:, 0:600]
  LAB_test = np.expand_dims(np.int32(BIG_test[:,600]), axis=1)

  #one-hot encoding for the labels
  LABELS_train = np.zeros((len(LAB_train), 3))
  LABELS_train[np.arange(len(LAB_train)), LAB_train[:,0]] = 1
  LABELS_test = np.zeros((len(LAB_test), 3))
  LABELS_test[np.arange(len(LAB_test)), LAB_test[:,0]] = 1

  return EMBEDD_train, LABELS_train, EMBEDD_test, LABELS_test   # splitting training pairs, test 

def main(_):

  # Create the model
  # Define input and output placeholders
  x = tf.placeholder(tf.float32, [None, 600])
  y_ = tf.placeholder(tf.float32, [None, 3])
  
  h_size = 10                            # hidden neurons in hidden layer
  W1 = tf.Variable(tf.zeros([600, h_size]))
  b1 = tf.Variable(tf.zeros([h_size]))
  W2 = tf.Variable(tf.zeros([h_size, 3]))
  b2 = tf.Variable(tf.zeros([3]))
  
  h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
  y = tf.matmul(h, W2) + b2
  '''
  W1 = tf.Variable(tf.zeros([600, 3]))
  b1 = tf.Variable(tf.zeros([3]))
  y = tf.matmul(x, W1) + b1
  '''
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  batch_xs, batch_ys, test_xs, test_ys = word_embeddding()  #Using regular gradient descent without batching i.e using the entire training set for each update
  
  # Train
  for _ in range(1000):
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model after each iteration
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  #print(sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys}))
  accuracy, result = sess.run([accuracy, y], feed_dict={x: test_xs, y_: test_ys})
  print(accuracy)
  print(result)

  index = result.argmax(axis =1)
  for i in range(len(index)):
    for j in range(3):
      if j == index[i]:
        result[i, j] = 1
      else:
        result[i, j] = 0

  print(index) 
  print(result)

  task1 = np.mean(result[:,0] == test_ys[:,0])
  task2 = np.mean(result[:,1] == test_ys[:,1])
  task3 = np.mean(result[:,2] == test_ys[:,2])
  print(task1)
  print(task2)
  print(task3)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Classification of relations')
  parser.add_argument('--pretrained', type=str, required=True, help='Pretrained model path')
  parser.add_argument('--dataset', type=str, required=True, help='Name of dataset BLESS, ROOT9')
  parser.add_argument('--split', type=float, required=True, help='Percentage of training examples [0 to 1]')
  parser.add_argument('--balance', type=float, required=True, help='1 for balanced datasets')
  parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
  args = parser.parse_args()
  tf.app.run(main=main)