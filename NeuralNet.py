__author__ = 'mcapizzi'

import tensorflow as tf
import pickle
import numpy as np
import math
import random

#TODO add autoencoder option

class NeuralNet:
    """
    builds neural network to be fed predicates
        number of classes ==> binary but with softmax layers
        :param embeddingClass ==> the embedding class from which the word2vecs can be loaded
        :param vectorSize ==> length of each individual embedding vector
        input ==> with each concatenated vector as a ROW
        :param learningRate ==> for gradient descent
        :param trainingEpochs ==> for gradient descent
        :param batchSize ==> for gradient descent
        :param displayStep = ???
        :param hiddenNodes ==> number of nodes in a single hidden layer
    """
    def __init__(self, embeddingClass, vectorSize, hiddenNodes, outputNodes, trainingEpochs, activationFunction, batchSize=None, learningRate=.001, displayStep=None):
        self.embeddingClass = embeddingClass
        self.vectorSize = vectorSize
        self.predicates = None      #TODO should this be []?
        self.vectors = []
        self.learningRate = learningRate
        self.trainingEpochs = trainingEpochs
        self.batchSize = batchSize
        self.displayStep = displayStep
        self.inputDimensions = 3 * vectorSize
        self.outputDimensions = outputNodes
        self.hiddenNodes = hiddenNodes
        self.activationFunction = activationFunction
        self.input = tf.placeholder("float", name="Input", shape=[None, self.inputDimensions])
        self.label = tf.placeholder("float", name="LabelDistribution", shape=[None, outputNodes])         #a distribution over T/F
        self.weights =  {}
        self.biases =   {}   #b initialized with standard randomized values
        self.session = tf.Session()
        self.init = tf.initialize_all_variables()

##############################################################################

    #pickle thing
    def pickle(self, thing, fname):
        f = open(fname + ".pickle", "wb")
        pickle.dump(thing, f)
        f.close()

    #TODO remove or fix - doesn't work
    #unpickle thing
    def unpickle(self, fname):
        f = open(fname + ".pickle", "rb")
        thing = pickle.load(fname)
        f.close()
        #return thing
        return thing


    #load embeddings
    def loadEmbeddings(self, fname):
        self.embeddingClass.loadModel(fname)


    #generate vector for a given predicate
    def getVector(self, predicate):
        #deconstruct copulars
        if "_" in predicate[1]:
            predPortion = predicate[1][3:]
            predicate = (predicate[0], "is", predPortion)

        if predicate[0]:
            #if the uppercase exists
            capture = self.embeddingClass.getVector(predicate[0])
            if capture is not None:
                subjectWord = capture
            #back off to trying lowercase
            else:
                subjectWord = self.embeddingClass.getVector(predicate[0].lower())
        else:
            subjectWord = None
        if predicate[1]:
            #if the uppercase exists
            capture = self.embeddingClass.getVector(predicate[1])
            if capture is not None:
                verbWord = capture
            #back off to trying lowercase
            else:
                verbWord = self.embeddingClass.getVector(predicate[1].lower())
        else:
            verbWord = None
        if predicate[2]:
            #if the uppercase exists
            capture = self.embeddingClass.getVector(predicate[2])
            if capture is not None:
                objectWord = capture
            #back off to trying lowercase
            else:
                objectWord = self.embeddingClass.getVector(predicate[2].lower())
                if objectWord is None:
                    objectWord = np.zeros(self.vectorSize)
        else:
            objectWord = np.zeros(self.vectorSize)
        if subjectWord is not None and verbWord is not None:
            return np.concatenate((subjectWord, verbWord, objectWord))


    #generate vectors for all predicates in NeuralNet class
    def getVectors(self):
        for predicate in self.predicates:
            #deconstruct copulars
            if "_" in predicate[1]:
                predPortion = predicate[1][3:]
                predicate = (predicate[0], "is", predPortion)

            if predicate[0]:
            #if the uppercase exists
                capture = self.embeddingClass.getVector(predicate[0])
                if capture is not None:
                    subjectWord = capture
                #back off to trying lowercase
                else:
                    subjectWord = self.embeddingClass.getVector(predicate[0].lower())
            else:
                subjectWord = None
            if predicate[1]:
                #if the uppercase exists
                capture = self.embeddingClass.getVector(predicate[1])
                if capture is not None:
                    verbWord = capture
                #back off to trying lowercase
                else:
                    verbWord = self.embeddingClass.getVector(predicate[1].lower())
            else:
                verbWord = None
            if predicate[2]:
                #if the uppercase exists
                capture = self.embeddingClass.getVector(predicate[2])
                if capture is not None:
                    objectWord = capture
                #back off to trying lowercase
                else:
                    objectWord = self.embeddingClass.getVector(predicate[2].lower())
                    if objectWord is None:
                        objectWord = np.zeros(self.vectorSize)
            else:
                objectWord = np.zeros(self.vectorSize)
            if subjectWord is not None and verbWord is not None:
                v = np.concatenate((subjectWord, verbWord, objectWord))
                self.vectors.append(v)




##############################################################################
    #TODO confirm this is done correctly
    #initialize weights
    def initializeParameters(self, useAutoEncoder=False, existingW1=None, existingW2=None):
        if useAutoEncoder:
            #load weights from existing variable
            self.weights["W1"] = existingW1
            self.weights["W2"] = existingW2
        else:
            self.weights["W1"] = tf.Variable(tf.random_normal([self.inputDimensions, self.hiddenNodes], mean=0, stddev=math.sqrt(float(6) / float(self.inputDimensions + self.outputDimensions + 1))), name="W1")
            self.weights["W2"] = tf.Variable(tf.random_normal([self.hiddenNodes, self.outputDimensions], mean=0, stddev=math.sqrt(float(6) / float(self.inputDimensions + self.outputDimensions + 1))), name="W2")

        self.biases["b1"] = tf.Variable(tf.random_normal([1, self.hiddenNodes], mean=0, stddev=math.sqrt(float(6) / float(self.inputDimensions + self.outputDimensions + 1))), name="b1")
        self.biases["b2"] = tf.Variable(tf.random_normal([1, self.outputDimensions], mean=0, stddev=math.sqrt(float(6) / float(self.inputDimensions + self.outputDimensions + 1))), name="b2")


##########

    #create feed-forward model
    def feedForward(self, inputX, secondBias):
        #z1
        z1 =    tf.add  (
                            tf.matmul   (
                                            inputX,
                                            self.weights["W1"]
                                        ),
                            self.biases["b1"]
                        ,name="Input->Hidden"
                        )

        #a1
        if self.activationFunction == "sigmoid":
            a1 = tf.nn.sigmoid(z1, name="HiddenActivation")
        elif self.activationFunction == "tanh":
            a1 = tf.nn.tanh(z1, name="HiddenActivation")
        elif self.activationFunction == "relu":
            a1 = tf.nn.relu(z1, name="HiddenActivation")
        else:   #default is tanh
            a1 = tf.nn.tanh(z1, name="HiddenActivation")

        #z2
        #with secondBias?
        if secondBias:
            z2 =    tf.add  (
                                tf.matmul   (
                                                a1,
                                                self.weights["W2"]
                                            ),
                                self.biases["b2"]
                            ,name="Hidden->Output")

        else:
            z2 =    tf.matmul   (
                                    a1,
                                    self.weights["W2"]
                                ,name="Hidden->Output")


        # #a2                                       #don't do this here (handled by cost
        # a2 = tf.nn.softmax(z2, "Softmax")
        #
        # #output
        # return a2
        return z2

    #TODO must generate the op
        #ffOp = feedforward(self.input, self.weights["W1"], self.weights["W2"], self.biases["b1"], self.biases["b2"], CHOOSE ACTIVATION)

##########
    #cost
        #output = feedforward
    def calculateCost(self, outputOp, labels):
        crossEntropy = tf.nn.softmax_cross_entropy_with_logits(outputOp, labels, name="CrossEntropy")
        loss = tf.reduce_mean(crossEntropy, name="Loss")
        return loss

    #TODO must generate op
        #costOp = calculateCost(ffOp, self.labels)

##########
    #optimizer
        #cost = calculateCost
    def train(self, costOp):
        return tf.train.GradientDescentOptimizer(self.learningRate).minimize(costOp)

    #TODO must generate op
        #gradientOp = train(costOp)

##########
    #predict
    def predict(self, feedforwardOp):
        softmax = tf.nn.softmax(feedforwardOp, name="Softmax")
        return softmax

    #TODO must generate op
        #predictOp = predict(ffOp)

##########
    #TODO include some way of training until convergence
        # if i > 0 and diff < .000001:
        #     break
        # else:
        #     #print
        #     print "iteration %s with average cost of %s and diff of %s" %(str(i),str(avgCost), str(diff))
    #run training
        #data = (vector, label)
        #optimizer = trainOp
    def runTraining(self, trainingData, gradientOp, costOp, isAutoEncoder=False):
        self.session.run(self.init)
        #initial average cost
        avgCost = 0

        #training cycle
        for epoch in range(self.trainingEpochs):
            #run gradient step
            if isAutoEncoder:
                self.session.run(gradientOp, feed_dict={self.input: trainingData[0], self.label: trainingData[0]})
            else:
                self.session.run(gradientOp, feed_dict={self.input: trainingData[0], self.label: trainingData[1]})
            #compute average cost
            if isAutoEncoder:
                newCost = self.session.run(costOp, feed_dict={self.input: trainingData[0], self.label: trainingData[0]})
            else:
                newCost = self.session.run(costOp, feed_dict={self.input: trainingData[0], self.label: trainingData[1]})
            #calculate diff from previous iteration
            diff = avgCost - newCost
            avgCost = newCost
            print("iteration %s with average cost of %s and diff of %" %(str(epoch+1), str(avgCost), str(diff)))


##########
    #close session
    def closeSession(self):
        self.session.close()


    #run prediction
    #use feedforward

#####################################################

def generateRandomPredicate(listOfSubjects, listOfVerbs, listOfObjects, allPredsList):
    #calculate percentage of preds without object
    noneObjsCount = float(listOfObjects.count(None)) / len(listOfObjects)
    #start with a predicate in list of predicates
    pred = allPredsList[0]
    #set up while loop to run until a pred NOT IN list of predicates appears
    while pred in allPredsList:
        random.shuffle(listOfSubjects)
        subjectWord = listOfSubjects[0]
        random.shuffle(listOfVerbs)
        verbWord = listOfVerbs[0]
        #determine whether there should be an object (according to odds of None in object) or if verb indicates copular
        if random.randint(0,1) > noneObjsCount and "_" not in verbWord:
            random.shuffle(listOfObjects)
            objectWord = listOfObjects[0]
        else:
            objectWord = None
        pred = (subjectWord, verbWord, objectWord)

    return pred

