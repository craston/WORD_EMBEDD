import gensim
import numpy as np



fname = "datasets/BLESS/BLESS_hyper-new4.txt"

model = gensim.models.KeyedVectors.load_word2vec_format('~/MoSIG/2016_data/S2/Summer_internship/Models/GoogleNews-vectors-negative300.bin', binary=True)  

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

print("HYPERNYMS")
print(np.array(v0).shape)
v0 = np.array(v0, dtype = np.float32)
v1 = np.array(v1, dtype = np.float32)

v_hyp = np.concatenate((v0,v1), axis=1)
labels_hyp = np.zeros(v_hyp.shape[0])
print (labels_hyp.size)
print (v_hyp.shape)

fname = "datasets/BLESS/BLESS_coord-new6.txt"

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

print("CO SIBLINGS")
print(np.array(v0).shape)
v0 = np.array(v0, dtype = np.float32)
v1 = np.array(v1, dtype = np.float32)

v_coord = np.concatenate((v0,v1), axis=1)
labels_coord = np.zeros(v_coord.shape[0])
print (labels_coord.size)
print (v_coord.shape)


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

print("RANDOM")
print(np.array(v0).shape)
v0 = np.array(v0, dtype = np.float32)
v1 = np.array(v1, dtype = np.float32)

v_rand = np.concatenate((v0,v1), axis=1)
labels_rand = np.zeros(v_rand.shape[0])
print (labels_rand.size)
print (v_rand.shape)

v_final = np.concatenate((v_hyp, v_rand, v_coord), axis=0)
labels_final = np.concatenate((labels_hyp, labels_rand, labels_coord), axis=0)
print("FINAL")
print(v_final.shape)
print(labels_final.shape) 
	

