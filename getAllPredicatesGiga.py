__author__ = 'mcapizzi'

import os
import Data as data
import Dependencies as dep
import time
import pickle
import itertools

#main method to get all predicates from gigaword files

#initialize variable for all predicates
# allPreds = []

d3StuckFile = ["200104"]
d3FinishedFile = ["199802", "200711", "200604", "200610", "199512", "200003", "200109", "200007", "200005", "200208", "200303", "201004", "199811", "200806", "200106", "200606", "199508", "200406", "200403", "200402", "199912", "201003", "200212", "200210", "200011", "201009"]
skipd3 = d3StuckFile + d3FinishedFile
d3ToSkip = map(lambda x: "gigaword_eng_5_d3_xin_eng_" + x, skipd3)

finishedDisk = []
toSkip = d3ToSkip

allDir = "/work/mcapizzi/Github/Content/src/main/resources/gigaword/"
allFiles = os.listdir(allDir)
for disk in itertools.ifilter(lambda x: x not in finishedDisk, allFiles):
    print disk
    for folder in os.listdir(allDir + disk + "/data/"):
        print folder
        for fileF in os.listdir(allDir + disk + "/data/" + folder):
            if fileF not in toSkip:
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
                    pf = open("Predicates/Gigaword/attempt02/" + disk + "_" + fileF + "-" + str(batch+1) + ".pickle", "wb")
                    raw = depClass.getSennaDeps(depClass.sentences[batch])
                    clean = depClass.cleanDeps("SENNA", raw)
                    preds = depClass.extractPredicates("SENNA", clean)
                    # [allPreds.append(p) for p in preds]
                    print("pickling batch %s of file %s" %(str(batch+1), fileF))
                    pickle.dump(preds, pf)
                    pf.close()
                    print "pickled batch %s of file %s" %(str(batch + 1), fileF)

# #if it gets through everthing without an error
# f2 = open("Predicates/Gigaword/ALL" + ".pickle", "wb")
# pickle.dump(allPreds, f2)
# f2.close()

