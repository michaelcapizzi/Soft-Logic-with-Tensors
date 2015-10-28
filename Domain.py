import numpy as np

#class for a domain

class Domain:

    #listOfElements = ["john", "chris", "tom"]
    #listOfPredictes = [("is_mathematician", ["john", "chris"])]
    def __init__(self, listOfElements, listOfUnaryPredicates = []):
        #domain and predicates
        self.elements = listOfElements
        self.elementLookUp = {}
        self.unaryPredicates = listOfUnaryPredicates
        self.unaryPredicateLookUp = {}
        self.sizeOfDomain = len(self.elements)
        self.domainMatrix = np.zeros((self.sizeOfDomain, self.sizeOfDomain))        #each row is a one-hot
        #truth conditions
        self.isTrue = np.array([1, 0]).transpose()
        self.isFalse = np.array([0, 1]).transpose()
        #TODO add connectives

    #build domain and lookup dictionary
    def buildDomain(self):
        for elem in range(self.sizeOfDomain):
            #add to lookup dictionary
            self.elementLookUp[self.elements[elem]] = elem
            #build one-hot vector
            oneHot = np.zeros(self.sizeOfDomain)
            oneHot[elem] = 1
            #add one-hot to domain matrix
            self.domainMatrix[elem] = oneHot


    #TODO build
    #build predicates and lookup dictionary
    def buildPredicates(self):
        for pred in range(len(self.unaryPredicates)):
            #build predicate matrix
            predMatrix = np.zeros((2, self.sizeOfDomain))
            for elem in self.elements:
                if elem in self.unaryPredicates[pred][1]:       #if the predicate applies to the element
                    #
                else:                                           #if the predicate does not apply to element
                    #


    #add an element to domain
    def addToDomain(self, element):

        #add to self.listOfElements
        self.elements.append(element)
        #update size of domain
        self.sizeOfDomain += 1
        #add to self.domainDictionary
        self.elementLookUp[element] = self.sizeOfDomain - 1
        #add to self.domainMatrix
        self.domainMatrix = np.insert(self.domainMatrix, self.domainMatrix.shape[1], 0, 1)      #add a column of zeros
        self.domainMatrix = np.insert(self.domainMatrix, self.domainMatrix.shape[0], 0, 0)      #add a row of zeros
        self.domainMatrix[self.domainMatrix.shape[0] - 1][self.domainMatrix.shape[1] - 1] = 1         #update one-hot vector

