__author__ = 'mcapizzi'

class Clusters:
    """
    clusters input vectors -- used for evaluation
        :param inputVectors ==> List[((subj, verb, obj), np.array)]

    """
    def __init__(self, inputVectors):
        self.vectors = inputVectors



    def kMeans(self, k):

