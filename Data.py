from nltk.parse.stanford import StanfordParser

__author__ = 'mcapizzi'

import re
import nltk as nl

class Data:

    """
    Takes a plain text file and sentence tokenizes, cleaning the input to ASCII.  allSentences can then be fed to annotator for dependency parsing
    """

    def __init__(self, file):
        self.file = file
        self.allSentences = []

    #sentence tokenize incoming lines
    def sentenceTokenize(self):
        for line in self.file:
            #decode in utf8
            decoded = line.decode("utf8")
            #sentence tokenize
            tokenizedSentences = nl.sent_tokenize(decoded)
            #flatten and add to allSentences
            [self.allSentences.append(i.rstrip()) for i in tokenizedSentences]

    #removes \n and html lines and titles and converts to ASCII
    def makeASCII(self):
        docRegex = r'[<>]'
        titleRegex = r'^\w+/.?$'
        unicodeRegex = r'\\u.*'
        self.allSentences = filter(  lambda x:   x != "" and
                                                not re.search(docRegex, x, re.U) and
                                                not re.search(titleRegex, x, re.U) and
                                                not re.search(unicodeRegex, x, re.U),
                                            self.allSentences)
        self.allSentences = [i.encode("ascii", "ignore") for i in self.allSentences]




