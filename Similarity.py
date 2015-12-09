__author__ = 'mcapizzi'

import itertools
import sys
rank_eval_dir = "/home/mcapizzi/rankeval-master/src/"
sys.path.append(rank_eval_dir)
from sentence.ranking import Ranking
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
        #returns three lists of sorted predicates
            #third list is [] for object None original
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
        original = cartesianProd[0]
        #remove original from cartesian product list
        del cartesianProd[0]

        #partition cart prod into three lists
        partitions = \
        [
            #(-, [1], [2])
            list(itertools.ifilter(lambda x: x[1] == original[1] and x[2] == original[2], cartesianProd)),
            #([0], -, [2])
            list(itertools.ifilter(lambda x: x[0] == original[0] and x[2] == original[2], cartesianProd)),
            #([0], [1], -)  #will be [] for None objects
            list(itertools.ifilter(lambda x: x[0] == original[0] and x[1] == original[1], cartesianProd)),
        ]

        #sort each partition by the cosSim to original
        partitionsSorted = \
        [
            #(-, [1], [2])
            list(reversed(sorted(partitions[0], key=lambda x: x[0][1]))),
            #([0], -, [2])
            list(reversed(sorted(partitions[1], key=lambda x: x[1][1]))),
            #([0], [1], -)  #will be [] for None objects
            list(reversed(sorted(partitions[2], key=lambda x: x[2][1])))
        ]

        self.predicatesRankedCosSim = partitionsSorted
        return partitionsSorted



    #ranks predicates by truth value of NN output
    def rankClosestPredicatesNN(self):
        allPreds = []
        for part in self.predicatesRankedCosSim:
            partPreds = []
            for pred in part:
                #filter out the cosSim scores
                if pred[2] is None:         #how to handle mapping None
                    predNone = pred[0:2]
                    justPredNone = tuple(map(lambda x: x[0], predNone))
                    justPred = (justPredNone[0], justPredNone[1], None)
                else:
                    justPred = tuple(map(lambda x: x[0], pred))
                #get likelihood of predicate in NN model
                likelihood = self.nnClass.getLikelihood(justPred)
                #append to list with ORIGINAL pred (including cosSim ==> to match other ranked list)
                partPreds.append((pred, likelihood[0][0]))
                # partPreds.append(pred)
            #sort
            sortedPreds = list(reversed(sorted(partPreds, key=lambda x: x[1])))
            allPreds.append(sortedPreds)

        self.predicatesRankedNN = allPreds
        return allPreds

    #evaluate rankings
        #gold = self.predicatesRankedCosSim
        #model = self.predicatesRankedNN
    def evaluateRanks(self, rankMetric):
        #generate indexed lists
        golds = []
        nnPredicteds = []
        for j in range(len(self.predicatesRankedCosSim)):
            #convert each rank to indices
            # if not self.predicatesRankedCosSim[2]:
            #     gold = map(lambda x: x[0], self.predicatesRankedCosSim[j][0:2])
            # else:
            gold = self.predicatesRankedCosSim[j]
            # if not self.predicatesRankedNN[2]:
            #     nnPredicted = map(lambda x: x[0][0], self.predicatesRankedNN[j][0:2])
            # else:
            nnPredicted = map(lambda x: x[0], self.predicatesRankedNN[j])
            goldIDX = list(range(len(gold)))
            nnPredictedIDX = []
            for i in goldIDX:
                nnPredictedIDX.append(gold.index(nnPredicted[i]))
            golds.append(goldIDX)
            nnPredicteds.append(nnPredictedIDX)
        #evaluate
        #set up to use rankeval
        goldRanks = [Ranking(g) for g in golds if g]
        nnRanks = [Ranking(n) for n in nnPredicteds if n]
        #list to hold final scores
        ranks = []
        for k in range(len(goldRanks)):
            if rankMetric == "kendallTau":
                ktScore = kendall_tau_set([nnRanks[k]], [goldRanks[k]])["tau"]
                ranks.append(ktScore)
            elif rankMetric == "MRR":   #mean reciprocal rank
                mrrScore = mrr([nnRanks[k]], [goldRanks[k]])["mrr"]
                ranks.append(mrrScore)
            elif rankMetric == "NDGC":  #normalized discounted cumulative gain
                ndgcScore = avg_ndgc_err([nnRanks[k]], [goldRanks[k]])["ndgc"]
                ranks.append(ndgcScore)
            else:   #defaults to MRR
                mrrScore = mrr([nnRanks[k]], [goldRanks[k]])["mrr"]
                ranks.append(mrrScore)
        if rankMetric == "kendallTau":
            self.kendallTau = ranks
            return ranks
        elif rankMetric == "MRR":
            self.MRR = ranks
            return ranks
        elif rankMetric == "NDGC":
            self.NDGC = ranks
            return ranks
        else:
            self.MRR = ranks
            return ranks


    def runAll(self, topN, rankMetric):
        self.getClosestPredicates(topN)
        self.rankClosestPredicatesCosSim()
        self.rankClosestPredicatesNN()
        ranks = self.evaluateRanks(rankMetric)
        return ranks



