__author__ = 'mcapizzi'

import NeuralNet as nn
import tensorflow as tf
import Similarity as s
import itertools

class Evaluation:
    """
    class for evaluating dataset
    :param positivePredicateList ==> list of predicates to be evaluated
    :param embeddingClass ==> class for vectors (can already have embeddings loaded)
    :param hiddenNodes = number of hidden nodes
    :param activationFunction = activation function ("sigmoid", "tanh", "relu")
    """

    def __init__(self, positivePredicateList, embeddingClass, hiddenNodes, activationFunction):
        self.predicates = positivePredicateList
        self.relevantPredicates = None
        self.embeddingClass = embeddingClass
        self.nnClass = nn.NeuralNet (
                                        embeddingClass=embeddingClass,
                                        vectorSize=embeddingClass.getVectorSize(),
                                        hiddenNodes=hiddenNodes,
                                        outputNodes=2,
                                        trainingEpochs=10,
                                        activationFunction=activationFunction,
                                        costFunction="crossEntropy",
                                        learningRate=None
                                    )
        self.parameterFile = "Variables/variables_NN_preds2_" + str(hiddenNodes) + "-" + activationFunction + "-crossEntropy-decayedLR-10iters"
        #similarity classes
        self.similarityClasses = []


    #finds predicates relevant to a subset (wordNetList)
    def findRelevantPredicates(self, wordNetList, pos):
        relevant = []
        for word in wordNetList:
            if pos == "n":
                matching = itertools.ifilter(lambda x: x[0].lower() == word.lower() or x[2] == word, self.predicates)
            elif pos == "v":
                matching = itertools.ifilter(lambda x: x[1].lower() == word.lower(), self.predicates)
            [relevant.append(pred) for pred in matching]
        self.relevantPredicates = relevant[:]
        return relevant



    def buildNN(self):
        #initialize all variables
        print("creating variables")
        self.nnClass.weights["W1"] = tf.Variable(tf.zeros([self.nnClass.inputDimensions, self.nnClass.hiddenNodes]))
        self.nnClass.weights["W2"] = tf.Variable(tf.zeros([self.nnClass.hiddenNodes, self.nnClass.outputDimensions]))
        self.nnClass.biases["b1"] = tf.Variable(tf.zeros([1, self.nnClass.hiddenNodes]))
        self.nnClass.biases["b2"] = tf.Variable(tf.zeros([1, self.nnClass.outputDimensions]))
        #override with loaded parameters
        print("loading variable weights")
        self.nnClass.loadVariables(self.parameterFile, variableName="Variable", targetName=self.nnClass.weights["W1"])    #W1 = Variable
        self.nnClass.loadVariables(self.parameterFile, variableName="W2", targetName=self.nnClass.weights["W2"])
        self.nnClass.loadVariables(self.parameterFile, variableName="Variable_1", targetName=self.nnClass.biases["b1"])    #b1 = Variable_1
        self.nnClass.loadVariables(self.parameterFile, variableName="b2", targetName=self.nnClass.biases["b2"])
        #build computational graph
        print("building computational graph")
        self.nnClass.buildComputationGraph()


    #get similarity rankings for each predicate
    def getSimilarityRankings(self, topN, rankMetric):
        if not self.relevantPredicates:
            for pred in self.predicates:
                #create similarity class
                print(pred)
                sim = s.Similarity(pred, self.nnClass)
                #run ranking evaluation
                sim.runAll(topN, rankMetric)
                #store similarity class
                self.similarityClasses.append(sim)
        else:
            for pred in self.relevantPredicates:
                #create similarity class
                print(pred)
                sim = s.Similarity(pred, self.nnClass)
                #run ranking evaluation
                sim.runAll(topN, rankMetric)
                #store similarity class
                self.similarityClasses.append(sim)


    #calculates average across all words
    def getAverageSimilarity(self, part, rankMetric):
        if rankMetric == "kendallTau":
            if part == "subj":
                justSubjSim = map(lambda x: x.kendallTau[0], self.similarityClasses)
                return sum(justSubjSim) / float(len(justSubjSim))
            elif part == "verb":
                justVerbSim = map(lambda x: x.kendallTau[1], self.similarityClasses)
                return sum(justVerbSim) / float(len(justVerbSim))
            elif part == "obj":
                justObjSim = map(lambda x: x[2], list(itertools.ifilter(lambda x: len(x) == 3, map(lambda x: x.kendallTau, self.similarityClasses))))
                return sum(justObjSim) / float(len(justObjSim))
        elif rankMetric == "MRR":
            if part == "subj":
                justSubjSim = map(lambda x: x.MRR[0], self.similarityClasses)
                return sum(justSubjSim) / float(len(justSubjSim))
            elif part == "verb":
                justVerbSim = map(lambda x: x.MRR[1], self.similarityClasses)
                return sum(justVerbSim) / float(len(justVerbSim))
            elif part == "obj":
                justObjSim = map(lambda x: x[2], list(itertools.ifilter(lambda x: len(x) == 3, map(lambda x: x.MRR, self.similarityClasses))))
                return sum(justObjSim) / float(len(justObjSim))
        elif rankMetric == "NGDC":
            if part == "subj":
                justSubjSim = map(lambda x: x.NDGC[0], self.similarityClasses)
                return sum(justSubjSim) / float(len(justSubjSim))
            elif part == "verb":
                justVerbSim = map(lambda x: x.NDGC[1], self.similarityClasses)
                return sum(justVerbSim) / float(len(justVerbSim))
            elif part == "obj":
                justObjSim = map(lambda x: x[2], list(itertools.ifilter(lambda x: len(x) == 3, map(lambda x: x.NDGC, self.similarityClasses))))
                return sum(justObjSim) / float(len(justObjSim))

#######################################################################################
#######################################################################################

