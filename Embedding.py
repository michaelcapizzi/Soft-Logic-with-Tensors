__author__ = 'mcapizzi'

from gensim import models
import itertools

class Embedding:
    #get the word2vec embeddings to be used
        #filePath = path to default w2v's

    def __init__(self, filePath="/home/mcapizzi/Google_Drive_Arizona/Programming/word2Vec/GoogleNews-vectors-negative300.bin.gz"):
        self.W2Vpath = filePath
        self.embeddingModel = None
        self.trainingSentences = []
        self.total_words = len(list(itertools.chain(*self.trainingSentences)))

#use GoogleNews vectors
    def useDefault(self):
        self.embeddingModel = models.Word2Vec.load_word2vec_format(fname=self.W2Vpath, binary=True, norm_only=True)


#train new vectors

    #add tokenized sentences
    def addTrainingSentences(self, listOfSentences):
        self.trainingSentences = listOfSentences


    #build a model with pre-loaded sentences
    def buildModel(self, window, min_count, workers, negative, size=100, ):
        self.embeddingModel = models.word2vec.Word2Vec(sentences=self.trainingSentences, size=size, window=window, min_count=min_count, workers=workers, negative=negative)
        #when all training is complete - saves memory
        self.embeddingModel.init_sims(replace=True)

    #train line by line in stream
    def trainLineByLine(self, stream, window, min_count, workers, negative, size=100):
        #intialize model
        self.embeddingModel = models.word2vec.Word2Vec(sentences=None, size=size, window=window, min_count=min_count, workers=workers, negative=negative)

        #TODO confirm this is done correctly using the same sentences both times (for build_vocab and train)
        for line in stream:
            self.embeddingModel.build_vocab(line.split(" "))        #TODO confirm how input is coming in
            self.embeddingModel.train(line.split(" "))

