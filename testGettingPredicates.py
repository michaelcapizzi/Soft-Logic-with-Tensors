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

###############################################################
#get raw Senna deps
deps = depClass.getSennaDeps(dataClass.allSentences[0:10])
###############################################################
#clean Senna deps
cleanDeps = depClass.cleanDeps("SENNA", deps)
#extract predicates
preds = depClass.extractPredicates("SENNA", cleanDeps)
#add to allPreds
# [allPreds.append(p) for p in depClass.extractedPredicates]
[allPreds.append(p) for p in preds]

f.close()


################
#attempt to multiprocess TODO - figure out how to use
#
# #method for the whole process
# def getDeps():
#     #open file
#     print ("working on file: " + fileName)
#     f = open("simpleWikipedia/" + fileName)
#     #make Data class
#     dataClass = data.Data(f)
#     #tokenize
#     dataClass.sentenceTokenize()
#     #clean
#     dataClass.makeASCII()
#     #make dependencies class
#     depClass = dep.Dependencies(dataClass.allSentences)
#     #get raw Senna deps
#     depClass.getSennaDeps()
#     #clean Senna deps
#     depClass.cleanDeps("SENNA")
#     #extract predicates
#     depClass.extractPredicates("SENNA")
#
#     # return depClass.extractedPredicates
#     [allPreds.append(pred) for pred in depClass.extractedPredicates]
#
# pool = multiprocessing.Pool(processes=4)
#
# #set up processes - one for each file
# [pool.apply(wholeProcess, args=(z,)) for z in os.listdir("simpleWikipedia")]
#

############################################

#pickle
f = open("Predicates/Test-extracted-" + time.strftime("%m_%d") + ".pickle", "wb")

pickle.dump(allPreds, f)

f.close()