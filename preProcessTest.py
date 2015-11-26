__author__ = 'mcapizzi'

import nltk
import itertools

f = open("sentenceSample.txt")
z = f.readlines()
f.close()

rawSentences = []
for line in z:
    sents = nltk.sent_tokenize(line)
    [rawSentences.append(s) for s in sents]
sentences = list(itertools.ifilterfalse(lambda x: x == "\n", rawSentences))


