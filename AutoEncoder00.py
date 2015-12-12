__author__ = 'mcapizzi'

import pickle
import Embedding as e
import NeuralNet as nn
import tensorflow as tf

#build an autoencoder whose weights will be used to initialize neural network

#create Embedding class
w2v = e.Embedding()
#load vectors
print ("loading word2vec")
w2v.loadModel("word2VecModels/simpleWiki_dim=200_win=5_mincount=5_negative=5.model.gz")
print ("finished loading word2vec")


#create NN class
    #autoencoder
    #300 hidden nodes
    #10 training epochs
    #tanh activation
    #least squares loss
    #decayed learning rate
testNN = nn.NeuralNet(embeddingClass=w2v, vectorSize=w2v.getVectorSize(),hiddenNodes=150, outputNodes=3 * w2v.getVectorSize(), trainingEpochs=10, activationFunction="relu", costFunction="RMSE", learningRate=None)


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


#initialize parameters
print("initializing parameters -- randomly")
testNN.initializeParameters(useAutoEncoder=False)


#build computational graph
print("build computational graph")
testNN.buildComputationGraph()

#####################################################

if __name__ == "__main__":

    #initialize variables
    # testNN.initializeVariables()
    testNN.session.run(tf.initialize_all_variables())       #TODO - must done manually --- why?


    #run training
    testNN.runTraining(isAutoEncoder=True)


    #save parameters
    testNN.saveVariables("Variables/variables_AutoEncoder_preds2_150-relu-loss-decayedLR-10iters")







