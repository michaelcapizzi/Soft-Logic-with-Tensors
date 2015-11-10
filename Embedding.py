__author__ = 'mcapizzi'

import gensim
import itertools

class Embedding:
    #get the word2vec embeddings to be used
        #filePath = path to default w2v's

    def __init__(self, filePath="/home/mcapizzi/Google_Drive_Arizona/Programming/word2Vec/GoogleNews-vectors-negative300.bin.gz"):
        self.W2Vpath = filePath
        self.embeddingModel = None
        self.trainingSentences = []
        self.total_words = len(list(itertools.chain(*self.trainingSentences)))


    def addTrainingSentences(self, listOfSentences):
        self.trainingSentences = listOfSentences


    def useDefault(self):
        self.embeddingModel = gensim.models.Word2Vec.load_word2vec_format(fname=self.W2Vpath, binary=True, norm_only=True)


    def buildModel(self, listOfSentences, size, window, min_count, workers, negative):
        self.embeddingModel = gensim.models.word2vec.Word2Vec(sentences=listOfSentences, size=size, window=window, min_count=min_count, workers=workers, negative=negative)


    def train(self, listOfSentences):
        #do I have to build_vocab?
        self.embeddingModel.train(sentences=listOfSentences, total_words=self.total_words)