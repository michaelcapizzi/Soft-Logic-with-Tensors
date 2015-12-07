__author__ = 'mcapizzi'

import pickle
import Embedding as e
import NeuralNet as nn
import tensorflow as tf
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
testNN = nn.NeuralNet(embeddingClass=w2v, vectorSize=w2v.getVectorSize(),hiddenNodes=300, outputNodes=3 * w2v.getVectorSize(), trainingEpochs=2, activationFunction="relu")

#loading true predicates
# f = open("Predicates/ALL-predicates-31994.pickle", "rb")
f = open("Predicates/FILTERED-predicates.pickle", "rb")
testNN.predicates = pickle.load(f)
f.close()

#loading false predicates
# f = open("Predicates/ALL-negative_predicates-31994.pickle", "rb")
f = open("Predicates/FILTERED-negative_predicates.pickle", "rb")
testNN.negPredicates = pickle.load(f)
f.close()


#build full dataset
print("building dataset")
testNN.buildDataset()


#initialize parameters
print("initializing parameters -- randomly")
testNN.initializeParameters(useAutoEncoder=False)


#build computational graph
print("build computational graph")
testNN.buildComputationGraph()

#initialize variables
# testNN.initializeVariables()
testNN.session.run(tf.initialize_all_variables())       #TODO - must done manually --- why?

#run training
# testNN.runTraining(isAutoEncoder=True))





