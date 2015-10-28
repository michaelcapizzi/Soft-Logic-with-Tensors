import numpy as np

#class for a domain

class Domain:

    def __init__(self, listOfElements):
        self.listOfElements = listOfElements
        self.sizeOfDomain = self.domainMatrix.shape[0]
        self.domainDictionary = {}
        self.domainMatrix = np.zeros((self.sizeOfDomain, self.sizeOfDomain))


    #build domain and lookup dictionary
    def buildDomain(self):
        for elem in range(self.sizeOfDomain):
            #add to lookup dictionary
            self.domainDictionary[self.listOfElements[elem]] = elem
            #build one-hot vector
            oneHot = np.zeros(self.sizeOfDomain)
            oneHot[elem] = 1
            #add one-hot to domain matrix
            self.domainMatrix[elem] = oneHot


    #add an element to domain
    def addToDomain(self, element):

        #add to self.listOfElements
        self.listOfElements.append(element)
        #add to self.domainDictionary
        self.domainDictionary[len(self.listOfElements) - 1] = len(self.listOfElements) - 1
        #add to self.domainMatrix
        np.insert(self.domainMatrix, self.domainMatrix.shape[1], 0, 1)      #add a column of zeros
        np.insert(self.domainMatrix, self.domainMatrix.shape[0], 0, 0)      #add a row of zeros
        self.domainMatrix[self.sizeOfDomain][self.sizeOfDomain] = 1

