__author__ = 'mcapizzi'

import os
import Data as data
import Dependencies as dep
import time
import pickle
import itertools

#main method to get all predicates from wikipedia articles

#initialize variable for all predicates
allPreds = []


finished = ["wiki_05", "wiki_08", "wiki_03", "wiki_07", "wiki_02", "wiki_06", "wiki_01", "wiki_10"]
stuck = ["wiki_00", "wiki_09"]
skip = finished + stuck
allFiles = os.listdir("simpleWikipedia")
toAnalyze = itertools.ifilterfalse(lambda x: x in skip, allFiles)

#iterate through wiki files
# for file in os.listdir("simpleWikipedia"):
for file in toAnalyze:
    print ("handling file " + file)
    pf = open("Predicates/" + file + ".pickle", "wb")
    #open file
    f = open("simpleWikipedia/" + file)
    #make Data class
    dataClass = data.Data(f)
    #tokenize
    print "tokenizing"
    dataClass.sentenceTokenize()
    #clean
    print "cleaning"
    dataClass.makeASCII()
    #make dependencies class
        #with chunkSize of 50
    print "building dependency class"
    depClass = dep.Dependencies(dataClass.allSentences)

    for batch in range(len(depClass.sentences)):
        raw = depClass.getSennaDeps(depClass.sentences[batch])
        clean = depClass.cleanDeps("SENNA", raw)
        preds = depClass.extractPredicates("SENNA", clean)
        [allPreds.append(p) for p in preds]
        print("pickling batch %s of file %s" %(str(batch+1), file))
        pickle.dump(preds, pf)
    print "pickled file %s" %file
    pf.close()

#if it gets through everthing without an error
f2 = open("Predicates/ALL-" + time.strftime("%m_%d") + ".pickle", "wb")
pickle.dump(allPreds, f2)
f2.close()

