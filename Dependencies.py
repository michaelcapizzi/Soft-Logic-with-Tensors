import practnlptools

__author__ = 'mcapizzi'

import Data
from practnlptools import tools
import StanfordDependencies
from nltk.parse import stanford

class Dependencies:
    """
    Generates the predicate tuples from dependencies for the sentences from a given Data class.
        Can use
            (1) practnlp (SENNA) dependency parser or
            (2) Stanford dependency parser
    """

    def __init__(self, sentences):
        self.sentences = sentences
        self.sennaAnnotator = practnlptools.tools.Annotator()
        self.stanfordParser = stanford.StanfordParser("/home/mcapizzi/Github/Semantics/stanford-parser-full-2014-08-27/stanford-parser.jar", "/home/mcapizzi/Github/Semantics/stanford-parser-full-2014-08-27/stanford-parser-3.4.1-models.jar")
        self.stanfordAnnotator = StanfordDependencies.get_instance(jar_filename="/home/mcapizzi/Github/Semantics/stanford-parser-full-2014-08-27/stanford-parser.jar", backend="subprocess")    #backend="jpype" will allow lemmatize  TODO figure out error ( 'edu.stanford.nlp.trees.TreeGraphNode' object has no attribute 'copyCount')
        self.sennaRawDependencies = []
        self.stanfordRawDependencies = []
        self.stanfordCleanDependencies = []


    #adds more sentences (from a later Data class)
    def addSentences(self, moreSentences):
        self.sentences.append(moreSentences)


    #get raw stanford dependencies for self.sentences in batch
    def getStanfordDeps(self):
        parses = [self.stanfordParser.raw_parse(sent)[0].pprint() for sent in self.sentences]
        rawDependencies = self.stanfordAnnotator.convert_trees(parses, include_punct=False)
        self.stanfordRawDependencies.append(rawDependencies)


    #TODO figure out why dep_parse values of dict are empty
    #get raw SENNA dependencies for self.sentences in batch
    def getSennaDeps(self):
        output = self.sennaAnnotator.getBatchAnnotations(self.sentences, dep_parse=True)
        rawDependencies = [output[i]["dep_parse"].split("\n") for i in range(len(output))]
        self.sennaRawDependencies.append(rawDependencies)


    #cleans dependencies into tuples for use
        #type = "Stanford" or "SENNA"
    def cleanDependencies(self, depType):
        if depType == "SENNA":
            for sentence in self.sennaRawDependencies:
                #TODO complete
                print ("none")
        else:
            for sentence in self.stanfordRawDependencies[0]:
                self.stanfordCleanDependencies.append(cleanStanfordDep(sentence))

##########################################################
##########################################################

    #cleans a raw Stanford dependency
    #token[0] = token index (at 1)
    #token[1] = token
    #token[2] = lemma
    #token[3] = course POStag
    #token[4] = fine-grained POStag
    #token[5] = dependency relation
    #token[6] = head
    #token[7] = dependency relation to head

def cleanStanfordDep(rawDep):
    #intitialize list
    cleanDep = []
    #renumbers to remove gaps in indexing
    rawDep.renumber()
    for token in rawDep:
        word = token[1]
        relation = token[7]
        if token[6] == 0:
            head = None           #TODO what to do if token is head of sentence?
        else:
            head = rawDep[token[6] - 1][1]
        cleanDep.append((word, relation, head))
    return cleanDep

#cleans a raw SENNA dependency