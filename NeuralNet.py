__author__ = 'mcapizzi'

import tensorflow as tf
import pickle
import numpy as np
import math

#TODO update this class to match examples from tensorflow-Examples

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
    def __init__(self, embeddingClass, vectorSize, learningRate, trainingEpochs, batchSize, displayStep, hiddenNodes, activationFunction):
        self.embeddingClass = embeddingClass
        self.vectorSize = vectorSize
        self.predicates = None      #TODO should this be []?
        self.vectors = []
        self.learningRate = learningRate,
        self.trainingEpochs = trainingEpochs,
        self.batchSize = batchSize,
        self.displayStep = displayStep,
        self.inputDimensions = 3 * vectorSize,
        self.hiddenNodes = hiddenNodes,
        self.activationFunction = activationFunction
        self.input = tf.placeholder("float", name="Input", shape=[None, self.inputDimensions]),
        self.label = tf.placeholder("float", name="LabelDistribution", shape=[None, 2])         #a distribution over T/F
        self.weights =  {   #W initialized with standard randomized values
                            "W1": tf.Variable(tf.random_normal([3 * vectorSize, hiddenNodes], mean=0, stddev=math.sqrt(float(6) / float(6 * vectorSize))), name="W1"),
                            "W2": tf.Variable(tf.random_normal([hiddenNodes, 2], mean=0, stddev=math.sqrt(float(6) / float(6 * vectorSize))), name="W2")
                        }
        self.biases =   {   #b initialized with standard randomized values
                            "b1": tf.Variable(tf.random_normal([1, hiddenNodes], mean=0, stddev=math.sqrt(float(6) / float(6 * vectorSize))), name="b1"),
                            "b2": tf.Variable(tf.random_normal([1, 2], mean=0, stddev=math.sqrt(float(6) / float(6 * vectorSize))), name="b2")
                        }
        self.session = tf.Session()
        self.init = tf.initialize_all_variables()

##############################################################################

    #pickle thing
    def pickle(self, thing, fname):
        f = open(fname + ".pickle", "wb")
        pickle.dump(thing, f)
        f.close()


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


    #generate vectors for predicates
    def getVectors(self, dimensions):
        for predicate in self.predicates:
            if predicate[0]:
                #if the uppercase exists
                capture = self.embeddingClass.getVector(predicate[0])
                if capture:
                    subjectWord = capture
                #back off to trying lowercase
                elif self.embeddingClass.getVector(predicate[0].lower()):
                    subjectWord = self.embeddingClass.getVector(predicate[0].lower())
            else:
                subjectWord = None
            if predicate[1]:
                #if the uppercase exists
                capture = self.embeddingClass.getVector(predicate[1])
                if capture:
                    verbWord = capture
                #back off to trying lowercase
                elif self.embeddingClass.getVector(predicate[1].lower()):
                    verbWord = self.embeddingClass.getVector(predicate[1].lower())
            else:
                verbWord = None
            if predicate[2]:
                #if the uppercase exists
                capture = self.embeddingClass.getVector(predicate[2])
                if capture:
                    objectWord = capture
                #back off to trying lowercase
                elif self.embeddingClass.getVector(predicate[2].lower()):
                    objectWord = self.embeddingClass.getVector(predicate[2].lower())
            else:
                objectWord = np.zeros(dimensions)
            if subjectWord and verbWord:
                v = np.concatenate((subjectWord, verbWord, objectWord))
                self.vectors.append(v)




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
    def runTraining(self, trainingData, gradientOp, costOp):
        self.session.run(self.init)
        #initial average cost
        avgCost = 0

        #training cycle
        for epoch in range(self.trainingEpochs):
            self.session.run(gradientOp, feed_dict={self.input: trainingData[0], self.label: trainingData[1]})
            #compute average cost
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
