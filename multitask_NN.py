

""" Binary classification using neural network
#USAGE
python multitask_NN.py --pretrained Models/GoogleNews-vectors-negative300.bin --dataset BLESS_ROOT9 --split 0.1
python multitask_NN.py --pretrained Models/GoogleNews-vectors-negative300.bin --dataset ROOT9 --split 0.2

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
	labels_rand.fill(2)                    # Class label for rand = 1

	#================splitting into training and test =================#
	
	#Balancing the pairs:
	if(args.balance == 1):
		least_number = min([len(labels_hyper), len(labels_coord), len(labels_rand)])
		print("least number = {0}".format(least_number))

		v_hyper = v_hyper[:least_number,:]
		v_coord = v_coord[:least_number,:]
		v_rand  = v_rand[:least_number,:]
		labels_hyper = labels_hyper[:least_number]
		labels_coord = labels_coord[:least_number]
		labels_rand  = labels_rand[:least_number]
		print("testing length = {0}".format(round(args.split*len(labels_hyper))))
	
	v_hyper_train      = v_hyper[:round(args.split*len(labels_hyper)),:]
	v_hyper_test  	   = v_hyper[round(args.split*len(labels_hyper))+1:,:]
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

	#========== TASK 1 (HYPER) ===================================#
	LABELS_train1 = np.zeros((len(LAB_train), 2))
	LABELS_train1[:,0] = LABELS_train[:,0]
	LABELS_train1[:,1] = 1 - LABELS_train[:,0]

	LABELS_test1 = np.zeros((len(LAB_test), 2))
	LABELS_test1[:,0] = LABELS_test[:,0]
	LABELS_test1[:,1] = 1 - LABELS_test[:,0]

	#========== TASK 2 (HYPER) ===================================#
	LABELS_train2 = np.zeros((len(LAB_train), 2))
	LABELS_train2[:,0] = LABELS_train[:,1]
	LABELS_train2[:,1] = 1 - LABELS_train[:,1]

	LABELS_test2 = np.zeros((len(LAB_test), 2))
	LABELS_test2[:,0] = LABELS_test[:,1]
	LABELS_test2[:,1] = 1 - LABELS_test[:,1]

	#========== TASK 3 (HYPER) ===================================#
	LABELS_train3 = np.zeros((len(LAB_train), 2))
	LABELS_train3[:,0] = LABELS_train[:,2]
	LABELS_train3[:,1] = 1 - LABELS_train[:,2]

	LABELS_test3 = np.zeros((len(LAB_test), 2))
	LABELS_test3[:,0] = LABELS_test[:,2]
	LABELS_test3[:,1] = 1 - LABELS_test[:,2]

	return (EMBEDD_train, EMBEDD_test, LABELS_train1, LABELS_test1, LABELS_train2, LABELS_test2, LABELS_train3, LABELS_test3)

def main(_):

	# Create the model
	# Define input and output placeholders
	x = tf.placeholder(tf.float32, [None, 600])
	y1_ = tf.placeholder(tf.float32, [None, 2])
	y2_ = tf.placeholder(tf.float32, [None, 2])
	y3_ = tf.placeholder(tf.float32, [None, 2])

	# Define model
	h_size = 10   														# 20 neurons in hidden layer
	shared_W = tf.Variable(tf.zeros([600, h_size]))
	shared_b = tf.Variable(tf.zeros([h_size]))
	y1_W = tf.Variable(tf.zeros([h_size, 2]))
	y1_b = tf.Variable(tf.zeros([2]))
	y2_W = tf.Variable(tf.zeros([h_size, 2]))
	y2_b = tf.Variable(tf.zeros([2]))
	y3_W = tf.Variable(tf.zeros([h_size, 2]))
	y3_b = tf.Variable(tf.zeros([2]))

	h = tf.nn.sigmoid(tf.matmul(x, shared_W) + shared_b)
	y1 = tf.matmul(h, y1_W) + y1_b
	y2 = tf.matmul(h, y2_W) + y2_b
	y3 = tf.matmul(h, y3_W) + y3_b

	# Define loss and optimizer
	cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y1_, logits=y1))
	cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y2_, logits=y2))
	cross_entropy3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y3_, logits=y3))

	cross_entropy  = cross_entropy1 + cross_entropy2 + cross_entropy3

	train_step1 = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy1)
	train_step2 = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy2)
	train_step3 = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy3)

	train_step  = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	batch_x, test_x, batch_y1, test_y1, batch_y2, test_y2, batch_y3, test_y3 = word_embeddding(args)  #Using regular gradient descent without batching i.e using the entire training set for each update

	# Train
	for _ in range(2000):
		_, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y1_: batch_y1, y2_: batch_y2, y3_: batch_y3})
		print("Task loss = {0}".format(loss))

		'''
		_, loss1 = sess.run([train_step1, cross_entropy1], feed_dict={x: batch_x, y1_: batch_y1, y2_: batch_y2, y3_: batch_y3})
		print("Task1 loss = {0}".format(loss1))

		_, loss3 = sess.run([train_step3, cross_entropy3], feed_dict={x: batch_x, y1_: batch_y1, y2_: batch_y2, y3_: batch_y3})
		print("Task3 loss = {0}".format(loss3))

		_, loss2 = sess.run([train_step2, cross_entropy2], feed_dict={x: batch_x, y1_: batch_y1, y2_: batch_y2, y3_: batch_y3})
		print("Task2 loss = {0}".format(loss2))
		'''
		# Test trained model for each task after each  iteration
		correct_prediction1 = tf.equal(tf.argmax(y1, 1), tf.argmax(y1_, 1))
		accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

		correct_prediction2 = tf.equal(tf.argmax(y2, 1), tf.argmax(y2_, 1))
		accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

		correct_prediction3 = tf.equal(tf.argmax(y3, 1), tf.argmax(y3_, 1))
		accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))

		acc1, acc2, acc3 = sess.run([accuracy1, accuracy2, accuracy3], feed_dict={x: test_x, y1_: test_y1, y2_: test_y2, y3_: test_y3})
		print("Task1 accuracy = {0}".format(acc1))
		print("Task2 accuracy = {0}".format(acc2))
		print("Task3 accuracy = {0}".format(acc3))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Multitask classification of relations')
	parser.add_argument('--pretrained', type=str, required=True, help='Pretrained model path')
	parser.add_argument('--dataset', type=str, required=True, help='Name of dataset BLESS, ROOT9')
	parser.add_argument('--split', type=float, required=True, help='Percentage of training examples [0 to 1]')
	parser.add_argument('--balance', type=int, required=True, help='1 for balanced else unbalanced')
	args = parser.parse_args()
	tf.app.run(main=main)
