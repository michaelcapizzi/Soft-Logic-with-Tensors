import tensorflow as tf
import numpy as np
import math
import random

vectorSize = 300
hiddenLayers = 3
hiddenDimensions = [10, 5, 3]
outputDimensions = 2

inputDimensions = vectorSize * 3
activationFunction = "tanh"
costFunction = "crossEntropy"   #"RMSE"

#add inputDimensions as hiddenLayer[0]
hiddenDimensions.insert(0, inputDimensions)
#add outputDimensions as hiddenLayer[+1]
hiddenDimensions.append(outputDimensions)

#placeholders
input = tf.placeholder("float", name="Input", shape=[None, inputDimensions])
label = tf.placeholder("float", name="Label", shape=[None, outputDimensions])

########################################
#generate variables in loop
weights = {}
biases = {}

#confirmed correct code!
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

#code to set up evaluation of variables
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

#code to test that loop above builds correct sizes
for v in weights.keys():
    print(v + " has a shape of " + str(weights[v].eval(sess).shape))

for b in biases.keys():
    print(b + " has a shape of " + str(biases[b].eval(sess).shape))

########################################

randomInput = np.random.rand(1,900).astype("float32")

#create feed-forward architecture in loop

#TODO test
def feedForwardGeneralized(inputX, numberOfLayers, weightsDict, biasesDict, activation):
    #code for one layer
    def oneLayer(layerInputX, layerNumber, layerWeightsDict, layerBiasesDict, layerActivation):
        z = tf.add(
                    tf.matmul(
                                layerInputX,
                                layerWeightsDict["W" + str(layerNumber + 1)]),
                    layerBiasesDict["b" + str(layerNumber + 1)],
                    name="Hidden" + str(layerNumber)
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


    #initialize input with original input
    intoLayer = inputX

    #loop through all layers
        #don't put activation on last layer!
    for i in range(numberOfLayers):
        if i != numberOfLayers - 1:  #in all cases except output layer
            print "layer " + str(i)
            intoLayer = oneLayer(input, i, weightsDict, biasesDict, activation)
        else:
            print "output layer"
            intoLayer = oneLayer(input, i, weightsDict, biasesDict, "none")

    return intoLayer