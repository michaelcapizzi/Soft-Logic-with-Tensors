__author__ = 'mcapizzi'

import itertools
import sys
rank_eval_dir = "/home/mcapizzi/rankeval-master/src/"
sys.path.append(rank_eval_dir)
from evaluation.ranking.set import kendall_tau_set
from evaluation.ranking.set import mrr
from evaluation.ranking.set import avg_ndgc_err


class Similarity:
    """
    class to run evaluation of similar vectors
    :param predicate ==> the predicate to evaluate
    :param nnClass ==> the neural network class to generate the likelihood
    """

    def __init__(self, predicate, nnClass):
        self.predicate = predicate
        # self.vector = None
        self.vector = nnClass.getVector(predicate)
        self.nnClass = nnClass
        self.closestPredicates = None
        self.predicatesRankedCosSim = None
        self.predicatesRankedNN = None
        self.kendallTau = None
        self.MRR = None
        self.NDGC = None


    def getVector(self):
        vec = self.nnClass.getVector(self.predicate)
        self.vector = vec
        return vec


    def getClosestPredicates(self, topN, makeNone=False):
        closest = self.nnClass.getClosestPredicate(self.vector.reshape((self.nnClass.vectorSize*3,)), topN, makeNone)
        self.closestPredicates = closest
        return closest


    # makes Cartesian product of all closestPredicates and then sorts by cosine similarity
    def rankClosestPredicatesCosSim(self):
        #get cartesian product of all combinations
        if self.closestPredicates[2] is None:        #if no object
            preds = self.closestPredicates[0:2]      #drop None
            cartesianProdNoNone = list(itertools.product(*preds))
            #add None back to each tuple
            cartesianProd = [(pred[0], pred[1], None) for pred in cartesianProdNoNone]
        else:
            cartesianProd = list(itertools.product(*self.closestPredicates))

        #extract the original predicate - the one with highest mean of cosSims - the first
        # original = cartesianProd[0]
        #remove original from cartesian product list
        del cartesianProd[0]

        #create (tuple, mean cosSim)
        if None in map(lambda x: x[2], cartesianProd):
            cartesianProdCondensed = [(tuple(map(lambda x: x[0], y[0:2])), sum(map(lambda x: x[1], y[0:2])) / float(2)) for y in cartesianProd]
        else:
            cartesianProdCondensed = [(tuple(map(lambda x: x[0], y)), sum(map(lambda x: x[1], y)) / float(3)) for y in cartesianProd]

        #return sorted list
        sortedPreds = list(reversed(sorted(cartesianProdCondensed, key=lambda x: x[1])))
        #add None back in
        sortedPredsPlusNone = []
        for p in sortedPreds:
            if len(p[0]) == 2:
                pNone = (p[0][0], p[0][1], None)
            sortedPredsPlusNone.append((pNone, p[1]))
        self.predicatesRankedCosSim = sortedPredsPlusNone
        return sortedPredsPlusNone


    #ranks predicates by truth value of NN output
    def rankClosestPredicatesNN(self):
        preds = []
        for p in map(lambda x: x[0], self.predicatesRankedCosSim):
            if len(p) == 2:
                p = (p[0], p[1], None)
            likelihood = self.nnClass.getLikelihood(p)
            preds.append((p, likelihood[0][0]))
        sortedPreds = list(reversed(sorted(preds, key=lambda x: x[1])))
        self.predicatesRankedNN = sortedPreds
        return sortedPreds

    #TODO figure out how to include rankeval stuff
    #evaluate rankings
        #gold = self.predicatesRankedCosSim
        #model = self.predicatesRankedNN
    def evaluateRanks(self, rankMetric):
        #convert each rank to indices
        gold = map(lambda x: x[0], self.predicatesRankedCosSim)
        nnPredicted = map(lambda x: x[0], self.predicatesRankedNN)
        goldIDX = list(range(len(gold)))
        nnPredictedIDX = []
        for i in goldIDX:
            nnPredictedIDX.append(gold.index(nnPredicted[i]))
        #evaluate
        #set up to use rankeval
        goldRank = [Ranking(goldIDX)]
        nnRank = [Ranking(nnPredictedIDX)]
        if rankMetric == "kendallTau":
            ktScore = kendall_tau_set(nnRank, goldRank)["tau"]
            self.kendallTau = ktScore
            return ktScore
        elif rankMetric == "MRR":   #mean reciprocal rank
            mrrScore = mrr(nnRank, goldRank)["mrr"]
            self.MRR = mrrScore
            return mrrScore
        elif rankMetric == "NDGC":  #normalized discounted cumulative gain
            ndgcScore = avg_ndgc_err(nnRank, goldRank)["ndgc"]
            self.NDGC = ndgcScore
            return ndgcScore
        else:   #defaults to MRR
            mrrScore = mrr(nnRank, goldRank)["mrr"]
            self.MRR = mrrScore
            return mrrScore




