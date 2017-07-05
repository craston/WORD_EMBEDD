

""" Binary classification using neural network
#USAGE
python binary_classification_NN.py --pretrained Models/GoogleNews-vectors-negative300.bin --dataset BLESS --Relation1 random --Relation2 coord --split 0.2
python binary_classification_NN.py --pretrained Models/GoogleNews-vectors-negative300.bin --dataset ROOT9 --Relation1 random --Relation2 coord --split 0.2

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import gensim
import numpy as np


# function which generates WORD vectors and returns training and test feature vectors
def word_embeddding(args):

	model = gensim.models.KeyedVectors.load_word2vec_format('{0}'.format(args.pretrained), binary=True) 
	fname = "datasets/{0}/{0}_{1}-new.txt".format(args.dataset, args.Relation1)

	with open(fname) as f:
	  R1 = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	R1 = [x.strip('\n') for x in R1] 
	R1 = [x.split('\t') for x in R1] 
	R10 = [x[0].split('-n')[0] for x in R1] 
	R11 = [x[2].split('-n')[0] for x in R1] 

	v0 = [model[x] for x  in R10]        # Generate vector of word1 of pair  (dimension = 300)
	v1 = [model[x] for x in R11]         # Generate vector of word2 of pair  (dimension = 300)

	# Converting to float32 numpy array
	v0 = np.array(v0, dtype = np.float32)         
	v1 = np.array(v1, dtype = np.float32)

	v_R1 = np.concatenate((v0,v1), axis=1) # Generating feature vector for word pair by concatenating the vectors (dimension = 600)
	labels_R1 = np.zeros(v_R1.shape[0], dtype=np.int)   # Class label for hypernym = 0


	fname = "datasets/{0}/{0}_{1}-new.txt".format(args.dataset, args.Relation2)

	with open(fname) as f:
	  R2 = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	R2 = [x.strip('\n') for x in R2] 
	R2 = [x.split('\t') for x in R2] 
	R20 = [x[0].split('-n')[0] for x in R2] 
	R21 = [x[2].split('-n')[0] for x in R2] 

	v0 = [model[x] for x  in R20]        # Generate vector of word1 of pair  (dimension = 300)
	v1 = [model[x] for x in R21]         # Generate vector of word2 of pair  (dimension = 300)

	v0 = np.array(v0, dtype = np.float32)
	v1 = np.array(v1, dtype = np.float32)

	v_R2 = np.concatenate((v0,v1), axis=1) # Generating feature vector for word pair by concatenating the vectors (dimension = 600)
	labels_R2 = np.empty(v_R2.shape[0], dtype=np.int)
	labels_R2.fill(1)                    # Class label for co-sibling = 1
	
	v_final = np.concatenate((v_R1, v_R2), axis=0)      # Merging all vectors
	labels_final = np.concatenate((labels_R1, labels_R2), axis=0)
	labels_final = np.expand_dims(labels_final, axis=1)

	BIG = np.concatenate((v_final, labels_final), axis=1)
	np.random.shuffle(BIG)                                          # Shuffling the dataset

	EMBEDD = BIG[:, 0:600]
	LAB = BIG[:,600]

	LAB = np.int32(LAB)
	LAB = np.expand_dims(LAB, axis=1)

	#one-hot encoding for the labels
	LABELS = np.zeros((len(LAB), 2))
	LABELS[np.arange(len(LAB)), LAB[:,0]] = 1
	return EMBEDD[:round(args.split*len(LAB)),:], LABELS[:round(args.split*len(LAB)),:], EMBEDD[round((1-args.split)*len(LAB))+1:,:], LABELS[round((1-args.split)*len(LAB))+1:,:]   # splitting training 4094 pairs, test 

def main(_):

  # Create the model
  # Define input and output placeholders
  x = tf.placeholder(tf.float32, [None, 600])
  y_ = tf.placeholder(tf.float32, [None, 2])

  # Define model
  h_size = 20   														# 20 neurons in hidden layer
  W1 = tf.Variable(tf.zeros([600, h_size]))
  b1 = tf.Variable(tf.zeros([h_size]))
  W2 = tf.Variable(tf.zeros([h_size, 2]))
  b2 = tf.Variable(tf.zeros([2]))
  
  h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
  y = tf.matmul(h, W2) + b2
  #y = tf.matmul(x, W1) + b1
  # Define loss and optimizer
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  batch_xs, batch_ys, test_xs, test_ys = word_embeddding(args)  #Using regular gradient descent without batching i.e using the entire training set for each update
  
  # Train
  for _ in range(1000):
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model after each iteration
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys}))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Classification of relations')
	parser.add_argument('--pretrained', type=str, required=True, help='Pretrained model path')
	parser.add_argument('--dataset', type=str, required=True, help='Name of dataset BLESS, ROOT9')
	parser.add_argument('--Relation1', type=str, required=True, help='hyper, cooord, random')
	parser.add_argument('--Relation2', type=str, required=True, help='hyper, cooord, random')
	parser.add_argument('--split', type=float, required=True, help='Percentage of training examples [0 to 1]')
	args = parser.parse_args()
	tf.app.run(main=main)
