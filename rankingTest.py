import tensorflow as tf
import Embedding as e
import pickle
import numpy as np

#create Embedding class
w2v = e.Embedding()
#load vectors
print ("loading word2vec")
w2v.loadModel("word2VecModels/simpleWiki_dim=200_win=5_mincount=5_negative=5.model.gz")
print ("finished loading word2vec")

#loading true predicates
f = open("Predicates/FILTERED-predicatesNoWiki.pickle", "rb")
posPredicates = pickle.load(f)
f.close()

#loading false predicates
f = open("Predicates/FILTERED-negative_predicatesNoWiki.pickle", "rb")
negPredicates = pickle.load(f)
f.close()

##########################
##########################
#######################################

#generate vector for a given predicate
def getVector(predicate):
    #deconstruct copulars
    if "_" in predicate[1]:
        predPortion = predicate[1][3:]
        predicate = (predicate[0], "is", predPortion)

    if predicate[0]:
        #if the uppercase exists
        capture = w2v.getVector(predicate[0])
        if capture is not None:
            subjectWord = capture
        #back off to trying lowercase
        else:
            subjectWord = w2v.getVector(predicate[0].lower())
    else:
        subjectWord = None
    if predicate[1]:
        #if the uppercase exists
        capture = w2v.getVector(predicate[1])
        if capture is not None:
            verbWord = capture
        #back off to trying lowercase
        else:
            verbWord = w2v.getVector(predicate[1].lower())
    else:
        verbWord = None
    if predicate[2]:
        #if the uppercase exists
        capture = w2v.getVector(predicate[2])
        if capture is not None:
            objectWord = capture
        #back off to trying lowercase
        else:
            objectWord = w2v.getVector(predicate[2].lower())
            if objectWord is None:
                objectWord = np.zeros(vectorSize)
    else:
        objectWord = np.zeros(vectorSize)
    if subjectWord is not None and verbWord is not None:
        return np.concatenate((subjectWord, verbWord, objectWord)).reshape((1,vectorSize*3))       #returns in appropriate shape

######################################
######################################

def manualCost(pos, neg):
    return np.max(np.array([0.0, 1 - pos - neg]))

#############################################

vectorSize = w2v.getVectorSize()
inputDimensions = 3 * vectorSize
outputDimensions = 1
hiddenNodes = 300
epochs = 10
learningRate = tf.train.exponential_decay(
        learning_rate=0.01,
        global_step= 1,
        decay_steps=50000,   #should be size of data: estimated at 50k
        decay_rate= 0.95,
        staircase=True
)

sess = tf.Session()
writer = tf.train.SummaryWriter("summary_logs", sess.graph_def)

#placeholders
inputPlaceholder = tf.placeholder("float", name="inputs", shape=[2, inputDimensions])

#variables
weights = {}
biases = {}

weights["W1"] = tf.Variable(tf.random_normal([inputDimensions, hiddenNodes], mean=0, stddev=np.sqrt(float(6) / float(inputDimensions + outputDimensions + 1))), name="W1")
biases["b1"] = tf.Variable(tf.random_normal([1, hiddenNodes], mean=0, stddev=np.sqrt(float(6) / float(inputDimensions + outputDimensions + 1))), name="b1")

weights["W2"] = tf.Variable(tf.random_normal([hiddenNodes, outputDimensions], mean=0, stddev=np.sqrt(float(6) / float(inputDimensions + outputDimensions + 1))), name="W2")
biases["b2"] = tf.Variable(tf.random_normal([1, outputDimensions], mean=0, stddev=np.sqrt(float(6) / float(inputDimensions + outputDimensions + 1))), name="b2")

####################################
####################################

#create feed-forward model
def feedForward(inputX):
    #z1
    z1 =    tf.add  (
            tf.matmul   (
                    inputX,
                    weights["W1"]
            ),
            biases["b1"]
            ,name="InputToHidden"
    )

    #a1
    a1 = tf.nn.sigmoid(z1, name="HiddenActivation")
    # a1 = tf.nn.tanh(z1, name="HiddenActivation")

    #z2
    z2 =    tf.add  (
                tf.matmul   (
                        a1,
                        weights["W2"]
                ),
                biases["b2"]
                ,name="HiddenToOutput")

    #a2
    a2 = tf.nn.sigmoid(z2, name="OutputActivation")
    # a2 = tf.nn.tanh(z2, name="HiddenActivation")

    #output
    #positive predicate, negative predicate
    return a2

ffOp = feedForward(inputPlaceholder)

#testing
pos = getVector(posPredicates[0])
neg = getVector(negPredicates[0])

testInput = np.array([pos, neg]).reshape((2,600))

sess.run(tf.initialize_all_variables())

#cost
costOp = tf.maximum(0.0, 1 - tf.reduce_sum(ffOp))
costSummary = tf.scalar_summary("cost", costOp)

#training
# gradientOp = tf.train.GradientDescentOptimizer(learningRate).minimize(costOp)
gradientOp1 = tf.train.GradientDescentOptimizer(learningRate).compute_gradients(costOp)
gradientOp2 = tf.train.GradientDescentOptimizer(learningRate).apply_gradients(gradientOp1)
# gradientSummary = tf.histogram_summary("gradient", gradientOp)

merged = tf.merge_all_summaries()

for i in range(epochs):
    for j in range(len(posPredicates)):
        posPred = posPredicates[i]
        negPred = negPredicates[i]
        posVector = getVector(posPred)
        negVector = getVector(negPred)
        #run gradient descent
        sess.run(gradientOp2, feed_dict={inputPlaceholder: np.array([posVector, negVector]).reshape((2,600))})
        if (i + j) % 500 == 0:
            print i
            preds, cost, summ = sess.run([ffOp, costOp, merged], feed_dict={inputPlaceholder: np.array([posVector, negVector]).reshape((2,600))})
            print(i, "pos-predicate output: ", preds[0])
            print(i, "neg-predicate output: ", preds[1])
            print(i, "manual cost: ", manualCost(preds[0], preds[1]))
            print(i, "tf cost: ", cost)
            writer.add_summary(summ, i)


