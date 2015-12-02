__author__ = 'mcapizzi'

import os
import pickle

#concatenates individual pickle files into one

#initialize accumulative list
allPreds = []

for file in os.listdir("Predicates/attempt02/"):
    if file.endswith("pickle"):
        #open file
        print file
        f = open("Predicates/attempt02/" + file, "rb")
        #create object
        ps = pickle.load(f)
        #close file
        f.close()
        #add to allPreds
        [allPreds.append(p) for p in ps]

#get number of preds
size = len(allPreds)

#repickle
f = open("Predicates/ALL-predicates-" + str(size) + ".pickle", "wb")
pickle.dump(allPreds, f)
f.close()
