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
        self.stanfordAnnotator = StanfordDependencies.get_instance(jar_filename="/home/mcapizzi/Github/Semantics/stanford-parser-full-2014-08-27/stanford-parser.jar", backend="subprocess")
        self.sennaRawDependencies = []
        self.stanfordRawDependencies = []


    #adds more sentences (from a later Data class)
    def addSentences(self, moreSentences):
        self.sentences.append(moreSentences)


    #get raw stanford dependencies for self.sentences in batch
    def getStanfordDeps(self):
        parses = [self.stanfordParser.raw_parse(sent)[0].pprint() for sent in self.sentences]
        rawDependencies = self.stanfordAnnotator.convert_trees(parses, include_punct=False, representation="collapsed")
        self.stanfordRawDependencies.append(rawDependencies)


    #TODO figure out why dep_parse values of dict are empty
    #get raw SENNA dependencies for self.sentences in batch
    def getSennaDeps(self):
        output = self.sennaAnnotator.getBatchAnnotations(self.sentences, dep_parse=True)
        rawDependencies = [output[i]["dep_parse"].split("\n") for i in range(len(output))]
        self.sennaRawDependencies.append(rawDependencies)


    #TODO build
    #cleans dependencies into tuples for use
        #type = "Stanford" or "SENNA"
    # def cleanDependencies(self, depType):
    #     if depType == "SENNA":
    #         for sentence in self.sennaRawDependencies:
    #             #
    #     else:
    #         for sentence in self.stanfordRawDependencies:
    #             #

