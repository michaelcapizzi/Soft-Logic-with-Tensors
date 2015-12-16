__author__ = 'mcapizzi'

import os
import Data as data
import Dependencies as dep
import time
import pickle
import itertools

#main method to get all predicates from gigaword files

#initialize variable for all predicates
allPreds = []

finished = []
stuck = []
skip = finished + stuck

allDir = "/work/mcapizzi/Github/Content/src/main/resources/gigaword/"
allFiles = os.listdir(allDir)
for disk in allFiles:
    print disk
    for folder in os.listdir(allDir + disk + "/data/"):
        print folder
        for fileF in os.listdir(allDir + disk + "/data/" + folder):
            if fileF not in skip:
                print "handling " + fileF
                #open file
                f = open(allDir + disk + "/data/" + folder + "/" + fileF)
                #make Data class
                dataClass = data.Data(f)
                #tokenize
                print "tokenizing " + fileF
                dataClass.sentenceTokenize()
                #clean
                print "cleaning " + fileF
                dataClass.makeASCII()
                #make dependencies class
                #with chunkSize of 50
                print "building dependency classes for " + fileF
                depClass = dep.Dependencies(dataClass.allSentences)

                for batch in range(len(depClass.sentences)):
                    pf = open("Predicates/Gigaword/" + disk + "_" + fileF + "-" + str(batch+1) + ".pickle", "wb")
                    raw = depClass.getSennaDeps(depClass.sentences[batch])
                    clean = depClass.cleanDeps("SENNA", raw)
                    preds = depClass.extractPredicates("SENNA", clean)
                    [allPreds.append(p) for p in preds]
                    print("pickling batch %s of file %s" %(str(batch+1), fileF))
                    pickle.dump(preds, pf)
                    pf.close()
                    print "pickled batch %s of file %s" %(str(batch + 1), fileF)

#if it gets through everthing without an error
f2 = open("Predicates/Gigaword/ALL" + ".pickle", "wb")
pickle.dump(allPreds, f2)
f2.close()

