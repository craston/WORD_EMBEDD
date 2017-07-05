
# WORD Embeddings
## Binary Classification
### Neural network
binary_classification_NN.py  (Binary classification using neural network) 

Classify between 

    * hypernyms v/s co-siblings (hyper -- coord)
    *. hypernyms v/s random  (hyper -- random)
    * random v/s co-siblings (rand -- coord)

Argument split to suggest percentage of word pairs to be used for training (--split 02 means 20% training and 80% test)
    
USAGE:
```
python binary_classification_NN.py --pretrained Models/GoogleNews-vectors-negative300.bin --dataset BLESS --Relation1 random --Relation2 coord --split 0.2
python binary_classification_NN.py --pretrained Models/GoogleNews-vectors-negative300.bin --dataset ROOT9 --Relation1 random --Relation2 coord --split 0.2
```
### Logistic Regression
binary_classification_reg.py  (Binary classification using logistic regression) 

Classify between 
    * hypernyms v/s co-siblings (hyper -- coord)
    * hypernyms v/s random  (hyper -- random)
    * random v/s co-siblings (rand -- coord)
    
Argument split to suggest percentage of word pairs to be used for training (--split 02 means 20% training and 80% test)
    
USAGE:
```
python binary_classification_reg.py --pretrained Models/GoogleNews-vectors-negative300.bin --dataset BLESS --Relation1 random --Relation2 coord --split 0.2
python binary_classification_reg.py --pretrained Models/GoogleNews-vectors-negative300.bin --dataset ROOT9 --Relation1 random --Relation2 coord --split 0.2
```