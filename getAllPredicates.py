__author__ = 'mcapizzi'

import os
import Data as data
import Dependencies as dep
import time
import pickle

#main method to get all predicates from wikipedia articles

#initialize variable for all predicates
allPreds = []

#iterate through wiki files
for file in os.listdir("simpleWikipedia"):
    print ("handling file " + file)
    #open file
    f = open("simpleWikipedia/" + file)
    #make Data class
    dataClass = data.Data(f)
    #tokenize
    dataClass.sentenceTokenize()
    #clean
    dataClass.makeASCII()
    #make dependencies class
    depClass = dep.Dependencies(dataClass.allSentences)
    #get raw Senna deps
    depClass.getSennaDeps()
    #clean Senna deps
    depClass.cleanDeps("SENNA")
    #extract predicates
    depClass.extractPredicates("SENNA")
    #add to allPreds
    [allPreds.append(p) for p in depClass.extractedPredicates]
    f.close()


#pickle
f = open("Predicates/extracted-" + time.strftime("%m_%d") + ".pickle", "wb")

pickle.dump(allPreds, f)

f.close()

################

def wholeProcess(file):
    #open file
    f = open("simpleWikipedia/" + file)
    #make Data class
    dataClass = data.Data(f)
    #tokenize
    dataClass.sentenceTokenize()
    #clean
    dataClass.makeASCII()
    #make dependencies class
    depClass = dep.Dependencies(dataClass.allSentences)
    #get raw Senna deps
    depClass.getSennaDeps()
    #clean Senna deps
    depClass.cleanDeps("SENNA")
    #extract predicates
    depClass.extractPredicates("SENNA")

    return depClass.extractedPredicates