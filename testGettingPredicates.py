__author__ = 'mcapizzi'

import Data as data
import Dependencies as dep
import time
import pickle
import multiprocessing

#to see how long it takes to get predicates from wikiSample.txt

#initialize variable for all predicates
allPreds = []

#open file
f = open("wikiSample.txt")
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


############################################

#pickle
f = open("Predicates/Test-extracted-" + time.strftime("%m_%d") + ".pickle", "wb")

pickle.dump(allPreds, f)

f.close()