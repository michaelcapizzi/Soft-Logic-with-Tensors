__author__ = 'mcapizzi'

from nltk.corpus import wordnet
import itertools

"""
access wordnet for synonyms, antonyms, and hypernyms
    :param pos ==> n, a, v
"""

def get_wn_synonyms(word, pos):
    syns = []
    #filter only by appropriate POS
    posSynsets = list(itertools.ifilter(lambda x: x.pos() == pos, wordnet.synsets(word)))
    for item in posSynsets:
        for l in item.lemmas():
            #remove multiple word entries
            if "_" not in l:
                syns.append(l.name())
    return syns


def get_wn_antonyms(word, pos):
    ants = []
    #filter only by appropriate POS
    posSynsets = list(itertools.ifilter(lambda x: x.pos() == pos, wordnet.synsets(word)))
    for item in posSynsets:
        for l in item.lemmas():
            for a in l.antonyms():
                #remove multiple word entries
                if "_" not in a:
                    ants.append(a.name())
    return ants


def get_wn_hypernyms(word, pos):
    hypers = []
    #filter only by appropriate POS
    posSynsets = list(itertools.ifilter(lambda x: x.pos() == pos, wordnet.synsets(word)))
    for item in posSynsets:
        for hr in item.hypernyms():
            for l in hr.lemmas():
                #remove multiple word entries
                if "_" not in l:
                    hypers.append(l.name())
    return hypers


def get_wn_hyponyms(word, pos):
    hypos = []
    #filter only by appropriate POS
    posSynsets = list(itertools.ifilter(lambda x: x.pos() == pos, wordnet.synsets(word)))
    for item in posSynsets:
        for ho in item.hyponyms():
            for l in ho.lemmas():
                #remove multiple word entries
                if "_" not in l:
                    hypos.append(l.name())
    return hypos