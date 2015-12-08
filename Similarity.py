__author__ = 'mcapizzi'

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
        self.closestPredicates = []
        self.rankedPredicates = []


    def getVector(self):
        vec = self.nnClass.getVector(self.predicate)
        self.vector = vec
        return vec


    def getClosestPredicates(self, vector, topN, makeNone=False):
        closest = self.nnClass.getClosestPredicate(vector, topN, makeNone)
        self.closestPredicates = closest
        return closest


    #makes Cartesian product of all closestPredicates and then sorts by cosine similarity
    # def rankPredicates(self):