__author__ = 'mcapizzi'

import gensim
import itertools

class Embedding:
    #get the word2vec embeddings to be used
        #filePath = path to default w2v's

    def __init__(self, trainingSentences, filePath="/home/mcapizzi/Google_Drive_Arizona/Programming/word2Vec/GoogleNews-vectors-negative300.bin.gz"):
        self.W2Vpath = filePath
        self.embeddingModel = gensim.models.word2vec.Word2Vec(sentences=None, size=100, window=8, min_count=1, workers=4, negative=5)
        self.trainingSentences = trainingSentences
        self.total_words = len(list(itertools.chain(*self.trainingSentences)))


    def useDefault(self):
        self.embeddingModel.load_word2vec_format(fname=self.W2Vpath, binary=True, norm_only=True)


    def train(self, listOfSentences):
        #do I have to build_vocab?
        self.embeddingModel.train(sentences=listOfSentences, total_words=self.total_words)