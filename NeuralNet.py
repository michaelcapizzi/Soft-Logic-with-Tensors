__author__ = 'mcapizzi'

import tensorflow as tf
import math

class NeuralNet:
    """
    builds neural network to be fed predicates
        input ==> 600 nodes, concatenation of subj, predicate, obj
        number of classes ==> binary but with softmax layers
        :param learningRate
        :param trainingEpochs
        :param batchSize
        :param displayStep = ???
        :param hiddenNodes ==> number of nodes in single hidden layer
    """
    def __init__(self, learningRate, trainingEpochs, batchSize, displayStep, inputDimensions, hiddenNodes, activationFunction):
        self.learningRate = learningRate,
        self.trainingEpochs = trainingEpochs,
        self.batchSize = batchSize,
        self.displayStep = displayStep,
        self.inputDimensions = inputDimensions,
        self.hiddenNodes = hiddenNodes,
        self.activationFunction = activationFunction
        self.x = tf.placeholder("float", [None, self.inputDimensions]),
        self.y = tf.placeholder("float", [None, 2])
        self.weights =  {
                            "W1": tf.Variable(tf.random_normal([inputDimensions, hiddenNodes])),
                            "W2": tf.Variable(tf.random_normal([hiddenNodes, 2]))
                        }
        self.biases =   {
                            "b1": tf.Variable(tf.random_normal([hiddenNodes])),
                            "b2": tf.Variable(tf.random_normal([2]))
                        }
        self.session = tf.Session()
        self.init = None



    #create feed-forward model
    def feedForward(self, inputX, secondBias, secondActivation):
        #z1
        z1 =    (
                    tf.add  (
                                tf.matmul   (
                                                inputX,
                                                self.weights["W1"]
                                            ),
                                self.biases["b1"]
                            )
                )

        #a1
        if self.activationFunction == "sigmoid":
            a1 = tf.nn.sigmoid(z1, name="HiddenLayer")
        elif self.activationFunction == "tanh":
            a1 = tf.nn.tanh(z1, name="HiddenLayer")
        elif self.activationFunction == "relu":
            a1 = tf.nn.relu(z1, name="HiddenLayer")
        else:
            a1 = tf.nn.tanh(z1, name="HiddenLayer")

        #z2
        #with secondBias?
        if secondBias:
            z2 =    (
                        tf.add  (
                                    tf.matmul   (
                                                    a1,
                                                    self.weights["W2"]
                                                ),
                                    self.biases["b2"]
                                )
                    )
        else:
            z2 =    (
                        tf.matmul   (
                                        a1,
                                        self.weights["W2"]
                                    )
                    )

        #a2
        if secondActivation:
            if self.activationFunction == "sigmoid":
                a2 = tf.nn.sigmoid(z2, name="Output")
            elif self.activationFunction == "tanh":
                a2 = tf.nn.tanh(z2, name="Output")
            elif self.activationFunction == "relu":
                a2 = tf.nn.relu(z2, name="Output")
            else:
                a2 = tf.nn.tanh(z2, name="Output")
        else:
            a2 = z2

        #output
        return a2


    #cost
        #output = feedforward
    def calculateCost(self, output):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, self.y))


    #optimizer
        #cost = calculateCost
    def trainOp(self, cost):
        return tf.train.GradientDescentOptimizer(self.learningRate).minimize(cost)


    #predict
        #output = feedforward
    def predictOp(self, output):
        return tf.argmax(output, 2)     #TODO confirm 2 will return both probabilities


    #initialize
    def initialize(self):
        self.init = tf.initialize_all_variables()


    #run training
        #data = (vector, label)
        #optimizer = trainOp
    def runTraining(self, trainingData, optimizer, cost):
        self.session.run(self.init)
        #training cycle
        for epoch in range(self.trainingEpochs):
            avgCost = 0
            batch = int(trainingData/self.batchSize)
            #iterate over each batch
            for i in range(batch):
                self.session.run(optimizer, feed_dict={self.x: trainingData[0], self.y: trainingData[1]})
                avgCost += self.session.run(cost, feed_dict={self.x: trainingData[0], self.y: trainingData[1]}) / batch
            #display logs per epoch step
            if epoch % self.displayStep == 0:
                print("Epoch:", "%04d" % (epoch + 1),"cost=", "{:9f}".format(avgCost))


    #run prediction
    #use feedforward
