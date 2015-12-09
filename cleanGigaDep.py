__author__ = 'mcapizzi'

import sys
import Dependencies as d
import pickle

# f = open(sys.argv[1])
f = open("/work/mcapizzi/test.txtaa")

#initiate an empty dependency class
depClass = d.Dependencies([])

#for each line
for line in f:
    #clean the dependency
    s = line.rstrip().split("__")
    if len(s) != 3:
        break
    else:
        print(s)
        #add to raw depdency data structure
        depClass.sennaRawDependencies.append((s[2], s[1], s[0]))

f.close()

print("cleaning dependencies")
#extract the necessary ones
preds = depClass.extractPredicates("SENNA", [depClass.sennaRawDependencies])

print("pickling")
f = open("Predicates/gigaPreds-" + sys.argv[1][-10:] + ".pickle", "wb")
pickle.dump(preds, f)
f.close()

print("finished")