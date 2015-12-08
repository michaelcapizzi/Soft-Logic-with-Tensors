__author__ = 'mcapizzi'

import pickle
import Embedding as e
import NeuralNet as nn
import tensorflow as tf

#use neural network to predict likelihood of predicate

#create Embedding class
w2v = e.Embedding()
#load vectors
print ("loading word2vec")
w2v.loadModel("word2VecModels/simpleWiki_dim=200_win=5_mincount=5_negative=5.model.gz")
print ("finished loading word2vec")


#create NN class
#300 hidden nodes
#10 training epochs
#tanh activation
#cross_entropy loss
testNN = nn.NeuralNet(embeddingClass=w2v, vectorSize=w2v.getVectorSize(),hiddenNodes=300, outputNodes=2, trainingEpochs=10, activationFunction="tanh", costFunction="crossEntropy", learningRate=None)


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


#build full dataset #TODO remove - not needed for prediction
# print("building dataset")
# testNN.buildDataset()


#initialize all variables
# testNN.initializeParameters(useAutoEncoder=False)
#build empty variables for restoring
testNN.weights["W1"] = tf.Variable(tf.zeros([testNN.inputDimensions, testNN.hiddenNodes]))
testNN.weights["W2"] = tf.Variable(tf.zeros([testNN.hiddenNodes, testNN.outputDimensions]))
testNN.biases["b1"] = tf.Variable(tf.zeros([1, testNN.hiddenNodes]))
testNN.biases["b2"] = tf.Variable(tf.zeros([1, testNN.outputDimensions]))


#then override with saved parameters
#TODO test to ensure this works
# testNN.loadVariables("Variables/variables_NN_tanh-crossEntropy-decayedLR-10itersTEST")
#otherwise load each separately
testNN.loadVariables("Variables/variables_NN_tanh-crossEntropy-decayedLR-10iters", variableName="Variable", targetName=testNN.weights["W1"])    #W1 = Variable
testNN.loadVariables("Variables/variables_NN_tanh-crossEntropy-decayedLR-10iters", variableName="W2", targetName=testNN.weights["W2"])
testNN.loadVariables("Variables/variables_NN_tanh-crossEntropy-decayedLR-10iters", variableName="Variable_1", targetName=testNN.biases["b1"])    #b1 = Variable_1
testNN.loadVariables("Variables/variables_NN_tanh-crossEntropy-decayedLR-10iters", variableName="b2", targetName=testNN.biases["b2"])



#build computational graph
print("build computational graph")
testNN.buildComputationGraph()


#ready to predict
#testNN.getLikelihood(predicate)

