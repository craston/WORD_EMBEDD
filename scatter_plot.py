# TSNE scatter plot 
# USAGE
# python scatter_plot.py --dataset BLESS --pretrained Models/GoogleNews-vectors-negative300.bin

import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
import gensim
import argparse

import sklearn
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
	fname = "datasets/{0}/{0}_hyper-new.txt".format(args.dataset)

	model = gensim.models.KeyedVectors.load_word2vec_format('{0}'.format(args.pretrained), binary=True)  

	with open(fname) as f:
		hyper = f.readlines()
	  # you may also want to remove whitespace characters like `\n` at the end of each line
	hyper = [x.strip('\n') for x in hyper] 
	hyper = [x.split('\t') for x in hyper] 
	hyper0 = [x[0].split('-n')[0] for x in hyper] 
	hyper1 = [x[2].split('-n')[0] for x in hyper] 
	 
	v0 = [model[x] for x  in hyper0]        # Generate vector of word1 of pair  (dimension = 300)
	v1 = [model[x] for x in hyper1]         # Generate vector of word2 of pair  (dimension = 300)

	# Converting to float32 numpy array
	v0 = np.array(v0, dtype = np.float32)         
	v1 = np.array(v1, dtype = np.float32)

	v_hyp = np.concatenate((v0,v1), axis=1) # Generating feature vector for word pair by concatenating the vectors (dimension = 600)
	labels_hyp = np.zeros(v_hyp.shape[0], dtype=np.int)   # Class label for hypernym = 0

	fname = "datasets/{0}/{0}_coord-new.txt".format(args.dataset)

	with open(fname) as f:
		hyper = f.readlines()
	  # you may also want to remove whitespace characters like `\n` at the end of each line
	hyper = [x.strip('\n') for x in hyper] 
	hyper = [x.split('\t') for x in hyper] 
	hyper0 = [x[0].split('-n')[0] for x in hyper] 
	hyper1 = [x[2].split('-n')[0] for x in hyper] 

	v0 = [model[x] for x  in hyper0]        # Generate vector of word1 of pair  (dimension = 300)
	v1 = [model[x] for x in hyper1]         # Generate vector of word2 of pair  (dimension = 300)

	v0 = np.array(v0, dtype = np.float32)
	v1 = np.array(v1, dtype = np.float32)

	v_coord = np.concatenate((v0,v1), axis=1) # Generating feature vector for word pair by concatenating the vectors (dimension = 600)
	labels_coord = np.empty(v_coord.shape[0], dtype=np.int)
	labels_coord.fill(1)                    # Class label for co-sibling = 1

	fname = "datasets/{0}/{0}_random-new.txt".format(args.dataset)

	with open(fname) as f:
	    hyper = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	hyper = [x.strip('\n') for x in hyper] 
	hyper = [x.split('\t') for x in hyper] 
	hyper0 = [x[0].split('-n')[0] for x in hyper] 
	hyper1 = [x[2].split('-n')[0] for x in hyper] 

	v0 = [model[x] for x  in hyper0]
	v1 = [model[x] for x in hyper1]

	v0 = np.array(v0, dtype = np.float32)
	v1 = np.array(v1, dtype = np.float32)

	v_rand = np.concatenate((v0,v1), axis=1)
	labels_rand = np.empty(v_rand.shape[0], dtype=np.int32)
	labels_rand.fill(2)                       # Class label for random = 2

	v_final = np.concatenate((v_hyp[0:699,:], v_rand[0:699,:], v_coord[0:699,:]), axis=0)      # Merging all vectors
	labels_final = np.concatenate((labels_hyp[0:699], labels_rand[0:699], labels_coord[0:699]), axis=0)
	labels_final = np.expand_dims(labels_final, axis=1)

	X = v_final
	y = np.int32(labels_final)

	digits_proj = TSNE(n_components=2).fit_transform(X)

	plt.scatter(digits_proj[:,0], digits_proj[:,1], c=y, cmap=plt.cm.get_cmap("jet", 3))
	plt.colorbar(ticks=range(3))
	plt.clim(0, 2)
	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Classification of relations')
	parser.add_argument('--pretrained', type=str, required=True, help='Path of pretrained model')
	parser.add_argument('--dataset', type=str, required=True, help='dataset name')
	args = parser.parse_args()
	main()
