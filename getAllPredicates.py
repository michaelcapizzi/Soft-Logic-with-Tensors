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


#iterate through wiki files
for file in [os.listdir("simpleWikipedia")[0]]:       #TODO if works, remove [0] so that it'll run for all files
    print ("handling file " + file)
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
    depClass = dep.Dependencies(dataClass.allSentences, 200)

    # print "setting up queue"
    #set up queue
    # q = multiprocessing.Queue()

    #define function to multiprocess
    def multiProcess(sentenceBatch):
        #get dependencies
        rawBatch = depClass.getSennaDeps(sentenceBatch)
        #debugging
            #print first dependency
        print rawBatch[0]
        #cleaning dependencies
        cleanBatch = depClass.cleanDeps("SENNA", rawBatch)
        #extracting predicates
        extractedPreds = depClass.extractPredicates("SENNA", cleanBatch)
        #debugging
            #print all predicates
        for i in range(len(extractedPreds)):
            print extractedPreds[i]

        return extractedPreds

    print "initializing processes"
    pool = multiprocessing.Pool(processes=4)
    print "starting"
    results = [pool.apply_async(multiProcess, args=(depClass.sentences[z],)) for z in range(len(depClass.sentences))]
    retrievedResults = [p.get() for p in results]
    #add to allPreds
    [allPreds.append(pred) for pred in retrievedResults]

print allPreds
#pickle
f = open("Predicates/extracted-" + time.strftime("%m_%d") + ".pickle", "wb")

pickle.dump(allPreds, f)

f.close()

