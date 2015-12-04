__author__ = 'mcapizzi'

class Clusters:
    """
    clusters input vectors -- used for evaluation
        :param inputVectors ==> List[((subj, verb, obj), np.array)]

    """
    def __init__(self, inputVectors):
        self.vectors = inputVectors


    # #clusters word vectors by kMeans
    # def kMeans(self, k):
    #
    # #clusters word vectors by Gaussian Mixture Models
    # def GMM(self, k):
    #
    # #clusters word vectors by cosine similarity

