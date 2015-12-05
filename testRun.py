__author__ = 'mcapizzi'

import pickle
import Embedding as e
import NeuralNet as nn
#test run of the whole process


#import predicates
# fPreds = open("Predicates/ALL-predicates-31994.pickle", "rb")
# preds = pickle.load(fPreds)
# fPreds.close()


#create Embedding class
w2v = e.Embedding()
#load vectors
print ("loading word2vec")
w2v.loadModel("word2VecModels/simpleWiki_dim=200_win=5_mincount=5_negative=5.model.gz")
print ("finished loading word2vec")


#create NN class
    #autoencoder
    #300 hidden nodes
    #1000 training epochs
    #RELU activation
testNN = nn.NeuralNet(embeddingClass=w2v, vectorSize=w2v.getVectorSize(),hiddenNodes=300, outputNodes=3 * w2v.getVectorSize(), trainingEpochs=1000, activationFunction="relu")

f = open("Predicates/ALL-predicates-31994.pickle", "rb")
testNN.predicates = pickle.load(f)
f.close()


#test predicate -> vector
print testNN.predicates[0]
print testNN.getVector(testNN.predicates[0])


