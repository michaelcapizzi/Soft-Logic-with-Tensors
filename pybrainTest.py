from pybrain import TanhLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

ds = ClassificationDataSet(1, nb_classes=2, class_labels=["dead", "alive"])


for i in range(100):
    if i < 50:
        ds.appendLinked(i, 0)
    else:
        ds.appendLinked(i, 1)

trainDS, testDS = ds.splitWithProportion(.25)

net = buildNetwork(trainDS.indim, 10, trainDS.outdim, bias=True, hiddenclass=TanhLayer)

trainer = BackpropTrainer(net, trainDS)

i = 0
for i in range(100):
    i+= 1
    trained = trainer.train()
    print(trained)

# trainer.trainUntilConvergence(verbose=True)

