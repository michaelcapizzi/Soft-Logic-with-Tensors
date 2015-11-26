__author__ = 'mcapizzi'

import nltk

f = open("sentenceSample.txt")
z = f.readlines()
f.close()

sentences = []
for line in z:
    sents = nltk.sent_tokenize(line)
    [sentences.append(s) for s in sents]


