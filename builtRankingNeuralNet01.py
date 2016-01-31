__author__ = 'mcapizzi'

import pickle
import Embedding as e
import RankingNeuralNet as nn
import tensorflow as tf

#build a neural network using the weights generated from autoencoder

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
#decayed learning rate
testNN = nn.RankingNeuralNet(embeddingClass=w2v, vectorSize=w2v.getVectorSize(),hiddenNodes=400, outputNodes=1, trainingEpochs=10, activationFunction="tanh", costFunction="crossEntropy", learningRate=None)


#loading true predicates
f = open("Predicates/FILTERED-predicatesNoWiki.pickle", "rb")
testNN.predicates = pickle.load(f)
f.close()

#loading false predicates
f = open("Predicates/FILTERED-negative_predicatesNoWiki.pickle", "rb")
testNN.negPredicates = pickle.load(f)
f.close()


#build full dataset
print("building dataset")
testNN.buildDataset()

#build empty variables for restoring
importedW1 = tf.Variable(tf.zeros([testNN.inputDimensions, testNN.hiddenNodes]))
importedb1 = tf.Variable(tf.zeros([1, testNN.hiddenNodes]))

#load variables into empties
testNN.loadVariables("Variables/variables_AutoEncoder_preds2_400-tanh-loss-decayedLR-10iters", variableName="W1", targetName=importedW1)
testNN.loadVariables("Variables/variables_AutoEncoder_preds2_400-tanh-loss-decayedLR-10iters", variableName="b1", targetName=importedb1)

#create all parameters for NN (including initializing W1 and b1 from Autoencoder)
testNN.initializeParameters(useAutoEncoder=True, existingW1=importedW1, existingB1=importedb1)

#build computational graph
print("build computational graph")
testNN.buildComputationGraph()

#####################################################

if __name__ == "__main__":

    #initialize variables
    # testNN.initializeVariables()
    testNN.session.run(tf.initialize_all_variables())       #TODO - must done manually --- why?


    #run training
    testNN.runTraining(isAutoEncoder=False)


    #save parameters
    testNN.saveVariables("Variables/rankingAttempt01")








