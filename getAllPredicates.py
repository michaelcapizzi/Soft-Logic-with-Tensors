__author__ = 'mcapizzi'

import os
import Data as data
import Dependencies as dep
import time
import pickle
import multiprocessing

#main method to get all predicates from wikipedia articles

#initialize variable for all predicates
allPreds = []

#
# #iterate through wiki files
# for file in os.listdir("simpleWikipedia"):
#     print ("handling file " + file)
#     #open file
#     f = open("simpleWikipedia/" + file)
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
#     #add to allPreds
#     [allPreds.append(p) for p in depClass.extractedPredicates]
#     f.close()


################
#attempt to multiprocess TODO - figure out how to use

q = multiprocessing.Queue()

#method for the whole process
def wholeProcess(fileName):
    #open file
    print ("working on file: " + fileName)
    f = open("simpleWikipedia/" + fileName)
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

    # return depClass.extractedPredicates
    # [allPreds.append(pred) for pred in depClass.extractedPredicates]
    q.put(depClass.extractedPredicates)


#initialize processes
ps = [multiprocessing.Process(target=wholeProcess, args=(z,)) for z in os.listdir("simpleWikipedia")]

#start
[p.start() for p in ps]

#stop
[p.join() for p in ps]

#add to allPreds
[[allPreds.append(pred) for pred in q.get()] for p in ps]

############################################

#pickle
f = open("Predicates/extracted-" + time.strftime("%m_%d") + ".pickle", "wb")

pickle.dump(allPreds, f)

f.close()

