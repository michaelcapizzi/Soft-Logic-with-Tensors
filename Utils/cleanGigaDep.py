__author__ = 'mcapizzi'

import sys
import Dependencies as d
import pickle

f = open(sys.argv[1])

#initiate an empty dependency class
depClass = d.Dependencies([])

#for each line
for line in f:
    #clean the dependency
    s = line.split("__")
    #add to raw depdency data structure
    depClass.sennaRawDependencies.append((s[2], s[1], s[0]))

f.close()

#extract the necessary ones
depClass.sennaCleanDependencies = depClass.extractedPredicates("SENNA", depClass.sennaRawDependencies)

f = open("Predicates/gigaPreds-" + sys.argv[1] + ".pickle")
pickle.dump(depClass.sennaRawDependencies, f)
f.close()