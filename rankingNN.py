import tensorflow as tf
import numpy as np
import math
import random

###################
#global parameters
###################
vectorSize = 300
hiddenLayers = 1
hiddenDimensions = [500]
outputDimensions = 2

inputDimensions = vectorSize * 3
activationFunction = "tanh"
costFunction = "crossEntropy"   #"RMSE"

###################
#placeholders
###################

#one placeholder for positive feedforward and one for negative feedforward
positiveInput = tf.placeholder("float", name="Input", shape=[None, inputDimensions])
negativeInput = tf.placeholder("float", name="Input", shape=[None, inputDimensions])
label = tf.placeholder("float", name="Label", shape=[None, outputDimensions])

###################/
#set up variables
###################
#add inputDimensions as hiddenLayer[0]
hiddenDimensions.insert(0, inputDimensions)
#add outputDimensions as hiddenLayer[+1]
hiddenDimensions.append(outputDimensions)

#generate variables in loop
weights = {}
biases = {}

for i in range(len(hiddenDimensions) - 1):    #+1 to treat the output layer as a "hidden layer"
    weights["W{0}".format(i + 1)] = tf.Variable(tf.random_normal(
            [hiddenDimensions[i], hiddenDimensions[i + 1]],     #hidden[i] x hidden[i + 1]
            mean=0,
            stddev=math.sqrt(float(6) / float(hiddenDimensions[i] + hiddenDimensions[-1] + 1))),
            name="W" + str(i + 1)
    )
    biases["b{0}".format(i + 1)] = tf.Variable(tf.random_normal(
            [1, hiddenDimensions[i + 1]],       #1 x hidden[0]
            mean=0,
            stddev=math.sqrt(float(6) / float(inputDimensions + outputDimensions + 1))),
            name="b" + str(i+1)
    )

########################################

# code to set up evaluation of variables
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

# code to test that loop above builds correct sizes

defaultGraph = tf.get_default_graph()

# for v in defaultGraph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
for v in defaultGraph.get_collection(tf.GraphKeys.VARIABLES):
    print v.name + " has a shape of " + str(v.eval(sess).shape)

# for v in tf.trainable_variables():
#     print v.name


########################################

randomInput = np.random.rand(1,900).astype("float32")

###################
#set up layers
###################
#create feed-forward architecture in loop

#code for one layer
def oneLayer(layerInputX, layerNumber, layerWeightsDict, layerBiasesDict, layerActivation):
    if layerActivation != "none":
        z = tf.add(
                tf.matmul(
                        layerInputX,
                        layerWeightsDict["W" + str(layerNumber + 1)]),
                layerBiasesDict["b" + str(layerNumber + 1)],
                name="Hidden" + str(layerNumber)
        )
    else:   #to label the output layer as such
        z = tf.add(
                tf.matmul(
                        layerInputX,
                        layerWeightsDict["W" + str(layerNumber + 1)]),
                layerBiasesDict["b" + str(layerNumber + 1)],
                name="Output"
        )

    if layerActivation == "sigmoid":
        a = tf.nn.sigmoid(z, name="sigmoidActivation" + str(layerNumber))
    elif layerActivation == "tanh":
        a = tf.nn.tanh(z, name="tanhActivation" + str(layerNumber))
    elif layerActivation == "relu":
        a = tf.nn.relu(z, name="reluActivation" + str(layerNumber))
    else:       #none
        a = tf.add(z,0, name="noActivation" + str(layerNumber))      #a hack just to get the node labelled on Tensor Board

    return a


#code for multiple layers
def feedForwardGeneralized(inputX, numberOfLayers, weightsDict, biasesDict, activation):
    #initialize input with original input
    intoLayer = inputX

    #loop through all layers
    #don't put activation on last layer!
    for i in range(numberOfLayers):
        if i != numberOfLayers - 1:  #in all cases except output layer
            print "layer " + str(i)
            intoLayer = oneLayer(intoLayer, i, weightsDict, biasesDict, activation)
        else:
            print "output layer"
            intoLayer = oneLayer(intoLayer, i, weightsDict, biasesDict, "none")

    return intoLayer


#must make one ffOp for positives
#must make another ffOp for negatives

#code for cost function
def rankingCost(positiveFF, negativeFF):
    return tf.maximum(0.0, 1 - positiveFF + negativeFF)

###################
#remaining logistics
###################
def initializeVars():
    return tf.initialize_all_variables().run()



#create optimizer
def createOptimizer(learningRate, opToMinimize):
    return tf.train.GradientDescentOptimizer(learningRate).minimize(opToMinimize)
#we need to use apply_gradients after manually determining the gradient of each variable in tuple of (gradientValue, variable)
    # for v in defaultGraph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    #     (v, [gradient])
    # return tf.train.GradientDescentOptimizer(learningRate).apply_gradients()


###################
#accuracy op
###################

#creates accuracy op
def createAccuracyOp(predictOp, labelPlaceholder):
    correct_prediction = tf.equal(tf.argmax(predictOp, 1), tf.argmax(labelPlaceholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)
    return correct_prediction, accuracy, accuracy_summary


###################
#set up summaries
###################

#creates summary for a value
#see https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html
#name = name of summary
#tensor = tensor to capture in summary
#type = summary type
def createSummary(name, tensor, type):
    if type == "scalar":
        return tf.scalar_summary(name, tensor)
    elif type == "histogram":
        return tf.histogram_summary(name, tensor)
    else: #default to "scalar"
        return tf.scalar_summary(name, tensor)


#if you make a summary for each node
def mergeAllSummaries():
    return tf.merge_all_summaries()


#creates a summary writer to use
def createSummaryWriter(logLocation, includeGraph_def=True):
    return tf.train.SummaryWriter(logLocation, includeGraph_def)


###################
#training
###################

#train
#tf.Session
#merged op
#feed_dict
## of epochs
#summary writer
#how often to report summary
def train(data, tfSession, mergedOp, accuracyOp, optimizer, trainFeed, testFeed, writer, summaryStep):
    for i in range(len(data)):
        if i % summaryStep == 0: #record summary data and current accuracy
            result = tfSession.run([mergedOp, accuracyOp], feed_dict=testFeed)
            summaryString = result[0]
            accuracyReport = result[1]
            writer.add_summary(summaryString, i)
            print("accuracy at step %s: %s" %(i, accuracyReport))
        else:   #continue training as normal
            tfSession.run(optimizer, feed=trainFeed)
    #print final accuracy
    print(accuracyOp.eval(testFeed))



