__author__ = 'mcapizzi'

import tensorflow as tf
import pickle
import numpy as np
import math
import random

#TODO add penalty term to RMSE cost function
#TODO determine convergence method -- perhaps use separate steps for gradient
    #https://www.tensorflow.org/versions/master/api_docs/python/train.html#Optimizer.minimize

class NeuralNet:
    """
    builds neural network to be fed predicates
        number of classes ==> binary but with softmax layers
        :param embeddingClass ==> the embedding class from which the word2vecs can be loaded
        :param vectorSize ==> length of each individual embedding vector
        :param learningRate ==> for gradient descent
        :param trainingEpochs ==> for gradient descent
        :param batchSize ==> for gradient descent
        :param outputDimensions
        :param hiddenNodes
        :param activationFunction
        :param costFunction
        :param hiddenNodes ==> number of nodes in a single hidden layer
        tensorflow documentation ==> https://www.tensorflow.org/versions/master/api_docs/python/index.html
    """
    def __init__(self, embeddingClass, vectorSize, hiddenNodes, outputNodes, trainingEpochs, activationFunction, costFunction="crossEntropy", batchSize=None, learningRate=.001):
        self.embeddingClass = embeddingClass
        self.vectorSize = vectorSize
        #textual data
        self.predicates = None
        self.negPredicates = None
        self.allPredicates = []
        self.predLabels = []
        self.skippedPredicates = []     #for predicates who have no vectors
        #NN input and label data
        self.predVectors = []           #only used to batch convert predicates
        #hyperparameters
        if not learningRate:            #if no learning rate is set, use exponential_decay
            self.learningRate = tf.train.exponential_decay(
                learning_rate=0.01,
                global_step= 1,
                decay_steps=50000,   #should be size of data: estimated at 50k
                decay_rate= 0.95,
                staircase=True
            )
        else:
            self.learningRate = learningRate
        self.trainingEpochs = trainingEpochs
        self.batchSize = batchSize
        self.inputDimensions = 3 * vectorSize
        self.outputDimensions = outputNodes
        self.hiddenNodes = hiddenNodes
        self.activationFunction = activationFunction
        self.costFunction = costFunction
        #parameters
        self.weights =  {}
        self.biases =   {}
        self.input = tf.placeholder("float", name="Input", shape=[None, self.inputDimensions])
        self.label = tf.placeholder("float", name="LabelDistribution", shape=[None, outputNodes])
        #computation graph
        self.init = tf.initialize_all_variables()           #TODO must be done manually ???
        self.ffOp = None
        self.costOp = None
        self.gradientOp = None
        self.predictOp = None
        #Tensorflow session
        self.session = tf.Session()


##############################################################################
#utils
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

########

    #TODO fix or dump -- doesn't work
    #load embeddings
    def loadEmbeddings(self, fname):
        self.embeddingClass.loadModel(fname)

########

    #generate random predicates
    def getNegativePredicates(self):
        negPredList = []
        numberOfPositivePreds = len(self.predicates)
        allSubjs = list(map(lambda x: x[0], self.predicates))
        allVerbs = list(map(lambda x: x[1], self.predicates))
        allObjs = list(map(lambda x: x[2], self.predicates))

        #iterate as many negative examples as positive
        for i in range(numberOfPositivePreds):
            negPred = generateNegativePredicate(allSubjs, allVerbs, allObjs, self.predicates)
            print("generating negative predicate " + str(i), negPred)
            negPredList.append(negPred)
        #add to predicate list
        # [self.negPredicates.append(np) for np in negPredList]
        #shuffle list
        # random.shuffle(self.negPredicates)
        return negPredList


    # #build allPredicates plus labels
    def buildDataset(self):
        labeledPos = [(pp, np.array([[1,0]])) for pp in self.predicates]
        labeledNeg = [(pn, np.array([[0,1]])) for pn in self.negPredicates]
        allPreds = labeledPos + labeledNeg
        random.shuffle(allPreds)
        self.allPredicates = map(lambda x: x[0], allPreds)
        self.predLabels = map(lambda x: x[1], allPreds)


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
            return np.concatenate((subjectWord, verbWord, objectWord)).reshape((1,self.vectorSize*3))       #returns in appropriate shape


    # #filter out predicates not in embedding
    def filterOutPreds(self, predList):
        predsToKeep = []
        for i in range(len(predList)):
            print("%s of %s predicates" %(str(i + 1), str(len(predList))))
            v = self.getVector(predList[i])
            if v is not None:
                predsToKeep.append(predList[i])
        return predsToKeep



    #generate vectors for all predicates in NeuralNet class
    # def getVectors(self):
    #     for predicate in self.predicates:
    #         #deconstruct copulars
    #         if "_" in predicate[1]:
    #             predPortion = predicate[1][3:]
    #             predicate = (predicate[0], "is", predPortion)
    #
    #         if predicate[0]:
    #         #if the uppercase exists
    #             capture = self.embeddingClass.getVector(predicate[0])
    #             if capture is not None:
    #                 subjectWord = capture
    #             #back off to trying lowercase
    #             else:
    #                 subjectWord = self.embeddingClass.getVector(predicate[0].lower())
    #         else:
    #             subjectWord = None
    #         if predicate[1]:
    #             #if the uppercase exists
    #             capture = self.embeddingClass.getVector(predicate[1])
    #             if capture is not None:
    #                 verbWord = capture
    #             #back off to trying lowercase
    #             else:
    #                 verbWord = self.embeddingClass.getVector(predicate[1].lower())
    #         else:
    #             verbWord = None
    #         if predicate[2]:
    #             #if the uppercase exists
    #             capture = self.embeddingClass.getVector(predicate[2])
    #             if capture is not None:
    #                 objectWord = capture
    #             #back off to trying lowercase
    #             else:
    #                 objectWord = self.embeddingClass.getVector(predicate[2].lower())
    #                 if objectWord is None:
    #                     objectWord = np.zeros(self.vectorSize)
    #         else:
    #             objectWord = np.zeros(self.vectorSize)
    #         if subjectWord is not None and verbWord is not None:
    #             v = np.concatenate((subjectWord, verbWord, objectWord))
    #             self.predVectors.append(v)

########

    #find closest word to a given vector
        #topN = number of matches to return
    def getClosestWord(self, vector, topN):
        return self.embeddingClass.embeddingModel.most_similar([vector], topn=topN)


    #find closest predicate to a given predicate vector
        #makeNone = boolean ==> try to make it easier to predict None obj
    def getClosestPredicate(self, predVector, topN, makeNone):
        # predVector = predVector.reshape((predVector.shape[1],))
        subjVec = predVector[:self.vectorSize]
        verbVec = predVector[self.vectorSize: self.vectorSize * 2]
        objVec = predVector[self.vectorSize * 2:]
        possibleSubjs = self.getClosestWord(subjVec, topN)
        possibleVerbs = self.getClosestWord(verbVec, topN)
        if makeNone:
            if objVec.sum() < (.0000001 * self.vectorSize):         #attempt to approximate None object more effectively
                possibleObjs = None
            else:
                possibleObjs = self.getClosestWord(objVec, topN)
        else:
            if np.all(objVec == np.zeros(self.vectorSize)):
                possibleObjs = None
            else:
                possibleObjs = self.getClosestWord(objVec, topN)
        possible = (possibleSubjs, possibleVerbs, possibleObjs)
        return possible


    #get vector similarity of two words
    def vectorSimV(self, word1, word2):
        return self.embeddingClass.embeddingModel.similarity(word1, word2)

########

    #save variables -- saves all variables
    def saveVariables(self, fname):
        saver = tf.train.Saver()
        saver.save(self.session, fname + ".ckpt")


    #load variable
        #  if variableName is None loads all variables from saved
        #  note: all variables to be loaded must be built with zeros first!
    def loadVariables(self, fname, variableName=None, targetName=None):
        if variableName is None:
            saver = tf.train.Saver()
            saver.restore(self.session, fname + ".ckpt")
        else:
            saver = tf.train.Saver({variableName:targetName})
            saver.restore(self.session, fname + ".ckpt")
#########

    def visualizeWeights(self, W1):
        staticW1 = W1.eval(session=self.session)
        for i in range(staticW1.shape[1]):
            return self.getClosestPredicate(staticW1[:,i].reshape((staticW1.shape[0],)), 1, makeNone=False)

##############################################################################
#variable setup
##############################################################################

    #TODO confirm this is done correctly
    #initialize weights
    def initializeParameters(self, useAutoEncoder=False, existingW1=None, existingB1=None):
        if useAutoEncoder:
            #load weights from existing variable
            self.weights["W1"] = existingW1
            self.biases["b1"] = existingB1
        else:
            self.weights["W1"] = tf.Variable(tf.random_normal([self.inputDimensions, self.hiddenNodes], mean=0, stddev=math.sqrt(float(6) / float(self.inputDimensions + self.outputDimensions + 1))), name="W1")
            self.biases["b1"] = tf.Variable(tf.random_normal([1, self.hiddenNodes], mean=0, stddev=math.sqrt(float(6) / float(self.inputDimensions + self.outputDimensions + 1))), name="b1")

        self.weights["W2"] = tf.Variable(tf.random_normal([self.hiddenNodes, self.outputDimensions], mean=0, stddev=math.sqrt(float(6) / float(self.inputDimensions + self.outputDimensions + 1))), name="W2")
        self.biases["b2"] = tf.Variable(tf.random_normal([1, self.outputDimensions], mean=0, stddev=math.sqrt(float(6) / float(self.inputDimensions + self.outputDimensions + 1))), name="b2")


##############################################################################
#computational graph
##############################################################################

    #create feed-forward model
    def feedForward(self, inputX, secondBias):
        #z1
        z1 =    tf.add  (
                            tf.matmul   (
                                            inputX,
                                            self.weights["W1"]
                                        ),
                            self.biases["b1"]
                        ,name="InputToHidden"
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
                            ,name="HiddenToOutput")

        else:
            z2 =    tf.matmul   (
                                    a1,
                                    self.weights["W2"]
                                ,name="HiddenToOutput")


        # #a2                                       #don't do this here (handled by cost
        # a2 = tf.nn.softmax(z2, "Softmax")
        #
        # #output
        # return a2
        return z2

##########

    #cost
        #output = feedforward
    def calculateCost(self, outputOp, labels, isAutoEncoder=False):
        if isAutoEncoder:
            if self.costFunction == "crossEntropy":
                crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(outputOp, labels, name="CrossEntropy")
                loss = tf.reduce_mean(crossEntropy, name="Loss")
            else:
                squaredError = tf.nn.l2_loss(outputOp - labels, name="SquaredError")
                # squaredError = tf.pow(outputOp - labels, 2, name="SquaredError")
                # squaredError = tf.add(
                #                         tf.nn.l2_loss(outputOp - labels, name="SquaredError")
                #                         tf.add  (
                #                                 self.weights["W1"].eval(session=self.session).sum(),
                #                                 self.weights["W2"].eval(session=self.session).sum()
                #                                 ),
                #                         name="RegularizationTerm")
                loss = tf.reduce_mean(squaredError, name="Loss")
        else:
            crossEntropy = tf.nn.softmax_cross_entropy_with_logits(outputOp, labels, name="CrossEntropy")
            loss = tf.reduce_mean(crossEntropy, name="Loss")
        return loss

##########

    #optimizer
        #cost = calculateCost
    def train(self, costOp):
        return tf.train.GradientDescentOptimizer(self.learningRate).minimize(costOp)

##########

    #predict
    def predict(self, feedforwardOp):
        softmax = tf.nn.softmax(feedforwardOp, name="Softmax")
        return softmax

# #############################################################################
# build ops
# #############################################################################

    def buildComputationGraph(self):
        #feedforward op
            #secondBias set to True
        self.ffOp = self.feedForward(self.input, secondBias=True)
        #costOp
        self.costOp = self.calculateCost(self.ffOp, self.label, isAutoEncoder=True)
        #gradientOp
        self.gradientOp = self.train(self.costOp)
        #predictOp
        self.predictOp = self.predict(self.ffOp)


    def initializeVariables(self):
        #initialize variables
        self.session.run(self.init)

# #############################################################################
# train
# #############################################################################

    #TODO add code for batch
    #run training
        #data = (vector, label)
        #optimizer = trainOp
        #topN = top 2 vector matches for debugging
    def runTraining(self, convergenceValue = .000001, isAutoEncoder=False, topN=1):

        #initial average cost
        avgCost = 0
        #training epoch
        for epoch in range(self.trainingEpochs):
            #stochastic gradient descent
            for i in range(len(self.allPredicates)):
                #run gradient step
                #the predicate to be fed in
                pred = self.allPredicates[i]
                #the vector for that predicate
                vector = self.getVector(pred)
                #reshape vector
                # vector = vector.reshape((1,vector.shape[0]))
                #training step
                if isAutoEncoder:
                    self.session.run(self.gradientOp, feed_dict={self.input: vector, self.label: vector})
                else:
                    self.session.run(self.gradientOp, feed_dict={self.input: vector, self.label: self.predLabels[i]})
                #compute average cost
                if isAutoEncoder:
                    newCost = self.session.run(self.costOp, feed_dict={self.input: vector, self.label: vector})
                else:
                    newCost = self.session.run(self.costOp, feed_dict={self.input: vector, self.label: self.predLabels[i]})
                #calculate diff from previous iteration
                diff = avgCost - newCost
                avgCost = newCost
                #debugging
                if (i + epoch) % 500 == 0:              #will ensure that different predicates appear for debugging at each iteration
                    print("predicate in: ", pred)
                    if isAutoEncoder:
                        print("predicate out: ", self.getClosestPredicate(self.session.run(self.ffOp, feed_dict={self.input: vector}).reshape((600,)),topN, True))
                    else:
                        print("label out: ", self.session.run(self.predictOp, feed_dict={self.input: vector}))
                    print("iteration %s, training instance %s: with average cost of %s and diff of %s" %(str(epoch+1), str(i + 1), str(avgCost), str(diff)))
                #determine if convergence -- ensure after first full iteration of data
                if epoch > 0 and abs(diff) < convergenceValue:
                    print("Convergence at iteration %s, training instance %s" %(str(epoch+1), str(i+1)))
                    break
        print("Training complete.")


##########
    #close session
    def closeSession(self):
        self.session.close()

#####################################################
#####################################################
#####################################################

#more utils

def generateNegativePredicate(listOfSubjects, listOfVerbs, listOfObjects, allPredsList):
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



