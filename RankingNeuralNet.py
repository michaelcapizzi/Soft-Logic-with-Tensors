__author__ = 'mcapizzi'

import tensorflow as tf
import pickle
import numpy as np
import math
import random

#TODO add penalty term to RMSE cost function
#https://www.tensorflow.org/versions/master/api_docs/python/train.html#Optimizer.minimize


class RankingNeuralNet:
    """
    builds neural network to be fed predicates (using ranking cost function)
        number of classes ==> one output, representing true probability (false = 1 - true)
        :param embeddingClass ==> the embedding class from which the word2vecs can be loaded
        :param vectorSize ==> length of each individual embedding vector
        :param learningRate ==> for gradient descent
        :param trainingEpochs ==> for gradient descent
        :param batchSize ==> for gradient descent
        :param outputDimensions --> in this case will be 1
        :param hiddenNodes
        :param activationFunction
        :param hiddenNodes ==> number of nodes in a single hidden layer
        tensorflow documentation ==> https://www.tensorflow.org/versions/master/api_docs/python/index.html
    """
    def __init__(self, embeddingClass, vectorSize, hiddenNodes, outputNodes, trainingEpochs, activationFunction, batchSize=None, learningRate=.001):
        self.embeddingClass = embeddingClass
        self.vectorSize = vectorSize
        #textual data
        self.predicates = None
        self.negPredicates = None
        self.allPredicates = None
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
        #parameters
        self.weights =  {}
        self.biases =   {}
        self.positiveInput = tf.placeholder("float", name="posInput", shape=[None, self.inputDimensions])
        self.negativeInput = tf.placeholder("float", name="negInput", shape=[None, self.inputDimensions])
        self.aePlaceholderIn = tf.placeholder("float", name="aeInput", shape=[None, self.inputDimensions])            #placeholder only to be used by autoencoder
        self.aePlaceholderOut = tf.placeholder("float", name="aeOutput", shape=[None, self.inputDimensions])            #placeholder only to be used by autoencoder
        #computation graph
        self.init = tf.initialize_all_variables()           #TODO must be done manually ???
        self.ffPosOp = None
        self.ffNegOp = None
        self.costOp = None
        self.costSummary = None
        self.gradientOp = None
        self.gradientSummary = None
        self.predictOp = None
        self.allSummaryOps = None
        #Tensorflow session
        self.session = tf.Session()
        #summary writer
        self.writer = tf.train.SummaryWriter("summary_logs", self.session.graph_def)


    ##############################################################################
    #utils
    ##############################################################################

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
    def buildDatasetAE(self):
        labeledPos = [(pp, np.array([[1,0]])) for pp in self.predicates]
        labeledNeg = [(pn, np.array([[0,1]])) for pn in self.negPredicates]
        allPreds = labeledPos + labeledNeg
        random.shuffle(allPreds)
        self.allPredicates = map(lambda x: x[0], allPreds)


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

    #TODO replace with generalized methods

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


        #a2
        if self.activationFunction == "sigmoid":
            a2 = tf.nn.sigmoid(z2, name="OutputActivation")
        elif self.activationFunction == "tanh":
            a2 = tf.nn.tanh(z2, name="OutputActivation")
        elif self.activationFunction == "relu":
            a2 = tf.nn.relu(z2, name="OutputActivation")
        else:   #default is tanh
            a2 = tf.nn.tanh(z2, name="OutputActivation")

        #output
        return a2
        # return z2

    ##########
    #cost

    #FOR USE IN AUTOENCODER
        #output = feedforward
    def calculateCostAE(self, outputOp, labels):
        squaredError = tf.nn.l2_loss(outputOp - labels, name="SquaredError")
        #TODO what does reduce_mean do here when `squaredError` is already a scalar?
        loss = tf.reduce_mean(squaredError, name="Loss")
        return loss

    ##########

    #FOR RANKING STEP
        #outputCorr = feedforward output of correct
        #outputIncorr = feedforward output of incorrect
    def calculateCostRanking(self, outputCorr, outputIncorr):
        return tf.maximum(0.0, 1 - outputCorr + outputIncorr, name="RankingCost")

    ##########

    #optimizer
    #cost = calculateCost
    def train(self, costOp):
        return tf.train.GradientDescentOptimizer(self.learningRate).minimize(costOp)

    ##########

    #predict
        #returns likelihood of True
    def predict(self, feedForwardOp):
        return feedForwardOp

    ############################################################################
    #build ops
    ############################################################################

    def buildComputationGraph(self, isAutoEncoder=False):
        if isAutoEncoder:
            #feedforward op
            self.ffPosOp = self.feedForward(self.aePlaceholder, secondBias=True)
            #cost op
            self.costOp = self.calculateCostAE(self.ffPosOp, self.aePlaceholder)
            self.costSummary = tf.histogram_summary("costAE", self.costOp)
            #gradient op
            self.gradientOp = self.train(self.costOp)
            #predict op
            self.predictOp = self.predict(self.ffPosOp)
            self.allSummaryOps = tf.merge_all_summaries()
        else:
            self.ffPosOp = self.feedForward(self.positiveInput, secondBias=True)
            self.ffNegOp = self.feedForward(self.negativeInput, secondBias=True)
            #costOp
            self.costOp = self.calculateCostRanking(self.ffPosOp, self.ffNegOp)
            self.costSummary = tf.scalar_summary("cost", self.costOp)
            #gradient op
            self.gradientOp = self.train(self.costOp)
            #predictOp
            self.predictOp = self.predict(self.ffPosOp)
            self.allSummaryOps = tf.merge_all_summaries()


    #TODO confirm this must be done manually
    def initializeVariables(self):
        #initialize variables
        self.session.run(self.init)

    #############################################################################
    #train
    #############################################################################

    #TODO add code for batch
    #run training
    def runTraining(self, convergenceValue = .000001, isAutoEncoder=False, topN=1):

        if not isAutoEncoder:
            #training epoch
            for epoch in range(self.trainingEpochs):
                #stochastic gradient descent
                for i in range(len(self.predicates)):
                    #run gradient step
                    #the predicate to be fed in
                    posPred = self.predicates[i]
                    negPred = self.negPredicates[i]
                    #the vector for that predicate
                    posVector = self.getVector(posPred)
                    negVector = self.getVector(negPred)
                    #reshape vector
                    # vector = vector.reshape((1,vector.shape[0]))
                    #training step
                    self.session.run(self.gradientOp, feed_dict={self.positiveInput: posVector, self.negativeInput: negVector})
                    #occasional reporting to summary and STDOUT
                    if (i + epoch) % 1000 == 0:              #will ensure that different predicates appear for debugging at each iteration
                        # summ, cost = self.session.run([self.allSummaryOps, self.costOp], feed_dict = {self.positiveInput: posVector, self.negativeInput: negVector})
                        cost = self.session.run([self.costOp], feed_dict = {self.positiveInput: posVector, self.negativeInput: negVector})
                        pos = self.session.run(self.ffPosOp, feed_dict={self.positiveInput: posVector})
                        neg = self.session.run(self.ffNegOp, feed_dict={self.negativeInput: negVector})
                        print("positive predicate in: ", posPred)
                        print("feedforward output of positive: ", pos)
                        print("negative predicate in: ", negPred)
                        print("feedforward output of negative: ", neg)
                        print("cost should be: %s" %str(1 - pos - neg))
                        print("tf output for cost: %s" %str(cost))
                        #write to summary
                        # self.writer.add_summary(summ, i)
        else:
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
                    self.session.run(self.gradientOp, feed_dict={self.aePlaceholderIn: vector, self.aePlaceholderOut: vector})
                    #calculate diff from previous iteration
                    if (i + epoch) % 1000 == 0:
                        newCost = self.session.run(self.costOp, feed_dict={self.aePlaceholderIn: vector, self.aePlaceholderOut: vector})
                        diff = avgCost - newCost
                        avgCost = newCost
                        print("predicate in: ", pred)
                        print("predicate out: ", self.getClosestPredicate(self.session.run(self.ffPosOp, feed_dict={self.positiveInput: vector}).reshape((600,)),topN, True))
                        print("iteration %s, training instance %s: with average cost of %s and diff of %s" %(str(epoch+1), str(i + 1), str(avgCost), str(diff)))
                    if epoch > 0 and abs(diff) < convergenceValue:
                        print("Convergence at iteration %s, training instance %s" %(str(epoch+1), str(i+1)))
                        break

        print("Training complete.")


    ##########

    #get likelihood of a predicate
    def getLikelihood(self, predicate):
        return self.session.run(self.predictOp, feed_dict={self.positiveInput: self.getVector(predicate)})

    ##########

    #close session
    def closeSession(self):
        self.session.close()

#####################################################
#####################################################
#####################################################

#more utils
#TODO fix ==> generating too many good predicates!
def generateNegativePredicate(listOfSubjects, listOfVerbs, listOfObjects, allPredsList):
    #calculate percentage of preds without object
    noneObjsCount = float(listOfObjects.count(None)) / len(listOfObjects)
    #start with a predicate in list of predicates
    pred = allPredsList[0]
    keep = random.uniform(0,1)
    #set up while loop to run until a pred NOT IN list of predicates appears
    #and to only keep every other predicate (since 50% of all predicates are bad)
    while pred in allPredsList or keep > .50:
        random.shuffle(listOfSubjects)
        subjectWord = listOfSubjects[0]
        random.shuffle(listOfVerbs)
        verbWord = listOfVerbs[0]
        #determine whether there should be an object (according to odds of None in object) or if verb indicates copular
        if "_" not in verbWord:
            random.shuffle(listOfObjects)
            objectWord = listOfObjects[0]
        else:
            objectWord = None
        pred = (subjectWord, verbWord, objectWord)
        keep = random.uniform(0,1)

    return pred



