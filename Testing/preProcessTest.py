__author__ = 'mcapizzi'

import Data as d
import nltk
import itertools

#from sentenceSample
f = open("sentenceSample.txt")
z = f.readlines()
f.close()

rawSampleSentences = []
for line in z:
    sents = nltk.sent_tokenize(line)
    [rawSampleSentences.append(s) for s in sents]
sampleSentences = list(itertools.ifilterfalse(lambda x: x == "\n", rawSampleSentences))

#from wikiSample
f = open("wikiSample.txt")
wiki = d.Data(f)
wiki.sentenceTokenize()
wiki.makeASCII()
#use wiki.allSentences
