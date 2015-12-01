import practnlptools

__author__ = 'mcapizzi'

from practnlptools import tools
import StanfordDependencies
from nltk.parse import stanford
import re

class Dependencies:
    """
    Generates the predicate tuples from dependencies for the sentences from a given Data class.
        Can use
            (1) practnlp (SENNA) dependency parser or
            (2) Stanford dependency parser
    """

    #TODO segment for easier use of big files

    def __init__(self, sentences):
        self.sentences = removeParen(sentences)
        self.sennaAnnotator = practnlptools.tools.Annotator()
        # self.stanfordParser = stanford.StanfordParser("/home/mcapizzi/Github/Semantics/stanford-parser-full-2014-08-27/stanford-parser.jar", "/home/mcapizzi/Github/Semantics/stanford-parser-full-2014-08-27/stanford-parser-3.4.1-models.jar")
        # self.stanfordParser = stanford.StanfordParser("stanford-parser-full-2014-08-27/stanford-parser.jar", "stanford-parser-full-2014-08-27/stanford-parser-3.4.1-models.jar")
        # self.stanfordAnnotator = StanfordDependencies.get_instance(jar_filename="/home/mcapizzi/Github/Semantics/stanford-parser-full-2014-08-27/stanford-parser.jar", backend="subprocess")    #backend="jpype" will allow lemmatize  TODO figure out error ( 'edu.stanford.nlp.trees.TreeGraphNode' object has no attribute 'copyCount')
        # self.stanfordAnnotator = StanfordDependencies.get_instance(jar_filename="stanford-parser-full-2014-08-27/stanford-parser.jar", backend="subprocess")    #backend="jpype" will allow lemmatize  TODO figure out error ( 'edu.stanford.nlp.trees.TreeGraphNode' object has no attribute 'copyCount')
        self.sennaRawDependencies = []
        self.sennaCleanDependencies = []
        self.stanfordRawDependencies = []
        self.stanfordCleanDependencies = []
        self.extractedPredicates = []


    #adds more sentences (from a later Data class)
    def addSentences(self, moreSentences):
        self.sentences.append(moreSentences)


    #get raw stanford dependencies for self.sentences in batch
    # def getStanfordDeps(self):
    #     parses = [self.stanfordParser.raw_parse(sent)[0].pprint() for sent in self.sentences]
    #     rawDependencies = self.stanfordAnnotator.convert_trees(parses, include_punct=False, representation="collapsed")
    #     self.stanfordRawDependencies = rawDependencies


    #get raw SENNA dependencies for self.sentences in batch
    def getSennaDeps(self):
        output = self.sennaAnnotator.getBatchAnnotations(self.sentences, dep_parse=True)
        rawDependencies = [output[i]["dep_parse"].split("\n") for i in range(len(output))]
        #add to list
        self.sennaRawDependencies = rawDependencies
        #also return list
        return rawDependencies


    #cleans dependencies into tuples for use
        #type = "Stanford" or "SENNA"
    def cleanDeps(self, depType):
        if depType == "SENNA":
            for sentence in self.sennaRawDependencies:
                self.sennaCleanDependencies.append(cleanSENNADep(sentence))
        else:
            for sentence in self.stanfordRawDependencies:
                self.stanfordCleanDependencies.append(cleanStanfordDep(sentence))


    #extracts predicates for use in semantic model
    def extractPredicates(self, depType):
        if depType == "SENNA":
            for sentence in self.sennaCleanDependencies:
                if extractPredicate(sentence):                  #if a sentence has predicates to extract
                    predicates = extractPredicate(sentence)
                    [self.extractedPredicates.append(predicates[j]) for j in range(len(predicates))]        #add each to self.predicates
        else:
            for sentence in self.stanfordCleanDependencies:
                if extractPredicate(sentence):                  #if a sentence has predicates to extract
                    predicates = extractPredicate(sentence)
                    [self.extractedPredicates.append(predicates[j]) for j in range(len(predicates))]        #add each to self.predicates
##########################################################
##########################################################

#removes any sentence with ( or )
    #since it can't be handled by SENNA and to guarantee Stanford has same length of sentences
def removeParen(listOfSentences):
    sentences = []
    for j in range(len(listOfSentences)):
        if "(" not in listOfSentences[j] and ")" not in listOfSentences[j]:
            sentences.append(listOfSentences[j])

    return [sentences[j].rstrip() for j in range(len(sentences))]


# cleans a raw Stanford dependency
    # token[0] = token index (at 1)
    # token[1] = token
    # token[2] = lemma
    # token[3] = course POStag
    # token[4] = fine-grained POStag
    # token[5] = dependency relation
    # token[6] = head
    # token[7] = dependency relation to head
def cleanStanfordDep(rawDep):
    #intitialize list
    cleanDep = []
    #renumbers to remove gaps in indexing
    rawDep.renumber()
    #reformat each tuple
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
def cleanSENNADep(rawDep):
    #initialize list
    cleanDep = []
    #reformat each tuple
    regex = r'(\w+)\((\w+)\.?-\d+, (\w+)\.?-\d+\)'     #\.? because some SENNA dependenices attach the period
    for token in rawDep:
        capture = re.search(regex, token)
        word = capture.group(3)
        relation = capture.group(1)
        if capture.group(2) == "ROOT":
            head = None             #TODO what to do if token is head of sentence?
        else:
            head = capture.group(2)
        cleanDep.append((word, relation, head))
    return cleanDep


#extracts appropriate predicates from sentence in a list
    #or returns None
def extractPredicate(cleanDep):
    #initialize variables
    predicates = {}
    predicateCounter = 0
    #iterate through each tuple
    for j in range(len(cleanDep)):
        #active sentence
        if cleanDep[j][1] == "nsubj" or cleanDep[j][1] == "xsubj":
            predicateCounter += 1
            predicates[predicateCounter] = (cleanDep[j][0], cleanDep[j][2], None)
            #look for object or copular
            for k in range(j, len(cleanDep)):
                #with object
                if cleanDep[k][1] == "dobj":
                    predicates[predicateCounter] = (predicates[predicateCounter][0], predicates[predicateCounter][1], cleanDep[k][0])
                    break
                #copular
                elif cleanDep[k][1] == "cop":
                    predicates[predicateCounter] = (predicates[predicateCounter][0], "is_" + predicates[predicateCounter][1], None)
                    break
            #ensure at least a subject and predicate
            if not predicates[predicateCounter][0] or not predicates[predicateCounter][1]:
                    predicates[predicateCounter] = None
        #passive sentence
        elif cleanDep[j][1] == "nsubjpass":
            predicateCounter += 1
            predicates[predicateCounter] = (None, cleanDep[j][2], cleanDep[j][0])
            #look for an agent
            for k in range(j, len(cleanDep)):
                if cleanDep[k][1] == "agent":
                    predicates[predicateCounter] = (cleanDep[k][0], predicates[predicateCounter][1], predicates[predicateCounter][2])
                    break
            #ensure an agent
            if not predicates[predicateCounter][0]:
                predicates[predicateCounter] = None

    # return predicates
    predicatesToReturn = []
    for pred in predicates.keys():
        if predicates[pred]:
            predicatesToReturn.append(predicates[pred])

    if predicatesToReturn:
        return predicatesToReturn
    else:
        return None

