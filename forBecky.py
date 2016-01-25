import re

###############################3
#sorts by score / highest -> lowest
def sortInsert(givenList, sentenceTuplePlusScore):
    if not givenList:
        givenList.append(sentenceTuplePlusScore)
    else:
        for i in range(len(givenList)):
            if sentenceTuplePlusScore[1] >= givenList[i][1]:
                givenList.insert(i, sentenceTuplePlusScore)
                return "sorted"
            else:
                next
        givenList.append(sentenceTuplePlusScore)
        return "sorted"

##################################

if __name__ == "__main__":

    #load NN
    import Predict02 as p


    #list to hold predicates and scores
    finalList = []
    skippedList = []

    #get predicates for Becky
    # f = open("/media/mcapizzi/data/nvn-tuples.txt", "r")
    f = open("nvn-tuples.txt", "r")

    # fKeep = open("/media/mcapizzi/data/nvn-tuplesKeep.txt", "w")
    fKeep = open("nvn-tuplesKeep.txt", "w")
    # fSkipped = open("/media/mcapizzi/data/nvn-tuplesSkipped.txt", "w")
    fSkipped = open("nvn-tuplesSkipped.txt", "w")
    fKeep.close()
    fSkipped.close()

    i = 0
    for line in f:
        i +=1
        if i % 1000 == 0:
            print i
        split = line.lstrip().split(" ")
        sentenceTuple = (split[2][1:-1], split[3][1:-1], split[4][1:-2])     #strips "" and keeps just tuple
        match0 = re.search(r'\W', sentenceTuple[0])
        match1 = re.search(r'\W', sentenceTuple[1])
        match2 = re.search(r'\W', sentenceTuple[2])
        if match0 or match1 or match2:
            # fSkipped = open("/media/mcapizzi/data/nvn-tuplesSkipped.txt", "a")
            fSkipped = open("nvn-tuplesSkipped.txt", "a")
            # print "skipping %s" %(str(sentenceTuple))
            skippedList.append(sentenceTuple)
            fSkipped.write(sentenceTuple[0] + " " + sentenceTuple[1] + " " + sentenceTuple[2] + "\n")
            fSkipped.close()
        elif p.testNN.getVector(sentenceTuple) is None:
            # fSkipped = open("/media/mcapizzi/data/nvn-tuplesSkipped.txt", "a")
            fSkipped = open("nvn-tuplesSkipped.txt", "a")
            # print "skipping %s" %(str(sentenceTuple))
            skippedList.append(sentenceTuple)
            fSkipped.write(sentenceTuple[0] + " " + sentenceTuple[1] + " " + sentenceTuple[2] + "\n")
            fSkipped.close()
        else:
            # print str(sentenceTuple)
            score = p.testNN.getLikelihood(sentenceTuple)[0][0]
            # fKeep = open("/media/mcapizzi/data/nvn-tuplesKeep.txt", "a")
            fKeep = open("nvn-tuplesKeep.txt", "a")
            fKeep.write(sentenceTuple[0] + " " + sentenceTuple[1] + " " + sentenceTuple[2] + "\t" + str(score) + "\n")
            fKeep.close()
            # print "scoring %s: %s" %(str(sentenceTuple), str(score))
            # unsorted.append((sentenceTuple, score))
            # print "sorting %s" %(str(sentenceTuple))
            sortInsert(finalList, (sentenceTuple, score))

    f.close()

    # fSorted = open("/media/mcapizzi/data/nvn-tuplesSorted.txt", "w")
    fSorted = open("nvn-tuplesSorted.txt", "w")
    # f2 = open("/media/mcapizzi/data/nvn-tuplesSkipped.txt", "w")
    #
    # print "writing skippedList"
    # for s in skippedList:
    #     f2.write(s[0] + " " + s[1] + " " + s[2] + "\n")
    # f2.close()

    print "writing sortedList"
    for i in range(len(finalList)):
        if i % 1000 == 0:
            print "writing %s of %s" %(str(i+1), str(len(finalList)))
        fSorted.write(finalList[i][0][0] + " " + finalList[i][0][1] + " " + finalList[i][0][2] + "\t" + str(finalList[i][1]) + "\n")

    fSorted.close()

    ####################


