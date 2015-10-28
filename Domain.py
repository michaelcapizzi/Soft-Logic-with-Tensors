import numpy as np

#class for a domain

class Domain:

    #listOfElements = ["john", "chris", "tom"]
    #listOfPredictes = [("is_mathematician", ["john", "chris"])]
    def __init__(self, listOfElements, listOfUnaryPredicates = []):
        self.elements = listOfElements
        self.elementLookUp = {}
        self.unaryPredicates = listOfUnaryPredicates
        self.unaryPredicateLookUp = {}
        self.sizeOfDomain = len(self.elements)
        self.domainMatrix = np.zeros((self.sizeOfDomain, self.sizeOfDomain))        #each row is a one-hot

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


    # #build predicates and lookup dictionary
    # def buildPredicates(self):
    #     for pred in range(len(self.predicates)):
    #         #build

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

