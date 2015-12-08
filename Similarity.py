__author__ = 'mcapizzi'

import itertools

class Similarity:
    """
    class to run evaluation of similar vectors
    :param predicate ==> the predicate to evaluate
    :param nnClass ==> the neural network class to generate the likelihood
    """

    def __init__(self, predicate, nnClass):
        self.predicate = predicate
        self.vector = None
        self.nnClass = nnClass
        self.closestPredicates = None
        self.predicatesRankedCosSim = None
        self.predicatesRankedNN = None


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
        self.predicatesRankedCosSim = sortedPreds
        return sortedPreds


    #rank predicates by truth value of NN output
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



