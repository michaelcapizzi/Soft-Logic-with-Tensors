import numpy as np

#elements in domain
    #one-hot vectors
john = np.array([1, 0, 0]).transpose()
chris = np.array([0, 1, 0]).transpose()
tom = np.array([0, 0, 1]).transpose()

domain = np.array([john, chris, tom])

#any unary predicate
    #column = john, chris, tom
    #row