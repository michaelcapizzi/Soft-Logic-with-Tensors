import numpy as np
import scipy.sparse as sp
import itertools

#class for a logic model
#TODO allow for loading predicates with uncertainty in calculations (i.e., not 1,0 and 0,1 for truth)
#TODO update with outer product (see Grefenstette)

class LogicModel:
    """
    creates a model that utilizes tensor-based application of first-order logic
    :param listOfElements = ["john", "chris", "tom"]
    :param dictionaryOfUnaryPredicates = {"is_mathematician": ["john", "chris"]}
    :param dictionaryOfBinaryPredicates = {"loves": [("john", "john"), ("chris", "john)]}
    :param use_sparse = boolean, whether to use sparse matrices or dense tensors
    """

    def __init__(self, listOfElements, dictionaryOfUnaryPredicates = {}, dictionaryOfBinaryPredicates = {}, use_sparse=False):
        self.use_sparse = use_sparse

        #domain
        self.elements = listOfElements
        self.elementLookUp = {}
        self.sizeOfDomain = len(self.elements)
        self.domainMatrix = np.zeros((self.sizeOfDomain, self.sizeOfDomain))        #each row is a one-hot
        #unary predicates
        self.unaryPredicateLookUp = self._normalizeUnaryPredicates(dictionaryOfUnaryPredicates)
        self.unaryPredicateMatrices = {}
        #binary predicates
        self.binaryPredicateLookUp = dictionaryOfBinaryPredicates
        self.binaryPredicateTensors = {}
        #truth conditions
        self.isTrue = np.array([1., 0.]).reshape((2,1))
        self.isFalse = np.array([0., 1.]).reshape((2,1))
        #connectives
        self.negConnect = np.array([
                                [0.,1.],
                                [1.,0.]
                            ])
        self.orConnect = np.array([                         #first row is first rank from left to right, top to bottom
                                    [1.,1.,0.,0.],
                                    [1.,0.,0.,1.]
                                ]).reshape((2,2,2))
        self.andConnect = np.array([
                                    [1.,0.,0.,1.],              #first row is first rank from left to right, top to bottom
                                    [0.,0.,1.,1.]
                                ]).reshape((2,2,2))
        self.conditionalConnect = np.array([                #first row is first rank from left to right, top to bottom
                                            [1.,0.,0.,1.],
                                            [1.,1.,0.,0.]
                                        ]).reshape((2,2,2))

    def _normalizeUnaryPredicates(self, dictionaryOfUnaryPredicates):
        normalized = {}
        for pred, elements in dictionaryOfUnaryPredicates.items():
            normalized[pred] = {}
            if isinstance(elements, dict):
                normalized[pred] = elements.copy()
            elif isinstance(elements, list):
                for item in elements:
                    if isinstance(item, tuple) and len(item) == 2 and (isinstance(item[1], float) or isinstance(item[1], int)):
                        # (element, probability)
                        normalized[pred][item[0]] = float(item[1])
                    else:
                        # element
                        normalized[pred][item] = 1.0
        return normalized

######################################################

#building the model

    #build entire model
    def buildAll(self):

        self.buildDomain()
        self.buildUnaryPredicates()
        self.buildBinaryPredicates()

    #build domain and lookup dictionary
    def buildDomain(self):
        for elem in range(self.sizeOfDomain):
            #add to lookup dictionary
            self.elementLookUp[self.elements[elem]] = elem
            #build one-hot vector
            # oneHot = np.zeros((self.sizeOfDomain, 1))
            oneHot = np.zeros(self.sizeOfDomain)
            oneHot[elem] = 1
            #add one-hot to domain matrix
            self.domainMatrix[:,elem] = oneHot


    #build unary predicates
    def buildUnaryPredicates(self):
        for pred in self.unaryPredicateLookUp.keys():
            if self.use_sparse:
                # Sparse: (2, N)
                # We can construct it as a list of data, row, col for coo_matrix or similar.
                # Since it's 2xN, dense is probably fine too, but let's stick to sparse for consistency.
                # Actually, 2 rows is very small. scipy.sparse is usually 2D.
                # Let's use lil_matrix for construction then convert to csc or csr.
                predMatrix = sp.lil_matrix((2, self.sizeOfDomain))
            else:
                #build predicate matrix
                predMatrix = np.zeros((2, self.sizeOfDomain))

            for elem in self.elements:
                idx = self.elementLookUp[elem]
                if elem in self.unaryPredicateLookUp[pred]:      #if the predicate applies to the element
                    prob = self.unaryPredicateLookUp[pred][elem]
                    if self.use_sparse:
                        predMatrix[:, idx] = np.array([prob, 1 - prob]).reshape(2, 1)
                    else:
                        predMatrix[:, idx] = np.array([prob, 1 - prob])
                else:                                           #if the predicate does not apply to element
                    if self.use_sparse:
                        predMatrix[:, idx] = self.isFalse # .T is (1,2) but slice is (2,).
                    else:
                        predMatrix[:, idx] = self.isFalse.flatten()
                    # Dense: predMatrix[:, idx] needs shape (2,). isFalse is (2,1).
                    # Sparse: slices assignment works if dimensions match.

            if self.use_sparse:
                self.unaryPredicateMatrices[pred] = predMatrix.tocsc()
            else:
                self.unaryPredicateMatrices[pred] = predMatrix


    def buildBinaryPredicates(self):
        for pred in self.binaryPredicateLookUp.keys():
            if self.use_sparse:
                # For sparse mode, we store (TrueMatrix, None)
                # TrueMatrix (sparse): stores P(True). Default 0.
                # Implicit P(False) = 1 - P(True).

                t_mat = sp.lil_matrix((self.sizeOfDomain, self.sizeOfDomain))

                # Iterate over known true pairs only
                for pair in self.binaryPredicateLookUp[pred]:
                    # Check if elements are in domain (safe-guard)
                    if pair[0] in self.elementLookUp and pair[1] in self.elementLookUp:
                         t_mat[self.elementLookUp[pair[1]], self.elementLookUp[pair[0]]] = 1.0

                self.binaryPredicateTensors[pred] = (t_mat.tocsc(), None)
            else:
                #build predicate tensor
                predTensor = np.zeros((2, self.sizeOfDomain, self.sizeOfDomain))
                self.binaryPredicateTensors[pred] = predTensor

                #get cartesian product
                cartProd = list(itertools.product(*[self.elements for i in [1,2]]))
                for pair in cartProd:
                    if pair in self.binaryPredicateLookUp[pred]:    #if the predicate applies to the ordered pair
                        #fill in true side of tensor (dim = 0) with a 1
                        #[0][obj][subj] = 1
                        predTensor[0][self.elementLookUp[pair[1]]][self.elementLookUp[pair[0]]] = 1
                        #false side of tensor (dim =1 ) will remain a 0
                    else:                                           #if the predicate doesn't apply to ordered pair
                        #true size of tensor (dim = 0) will remain a 0
                        #fill in false side of tensor (dim = 1) with a 1
                        predTensor[1][self.elementLookUp[pair[1]]][self.elementLookUp[pair[0]]] = 1

######################################################

#modifying the world
#TODO update to handle binary predicates!

    #TODO update to handle added elements in binary predicates
    #add an element to domain and necessary predicates
        #tupleToAdd => (element, [listOfPredicates])
    def addToDomain(self, tupleToAdd):
    #domain
        #add to self.listOfElements
        self.elements.append(tupleToAdd[0])
        #update size of domain
        self.sizeOfDomain += 1
        #add to self.domainDictionary
        self.elementLookUp[tupleToAdd[0]] = self.sizeOfDomain - 1
        #add to self.domainMatrix
        self.domainMatrix = np.insert(self.domainMatrix, self.domainMatrix.shape[1], 0, 1)      #add a column of zeros
        self.domainMatrix = np.insert(self.domainMatrix, self.domainMatrix.shape[0], 0, 0)      #add a row of zeros
        self.domainMatrix[self.domainMatrix.shape[0] - 1][self.domainMatrix.shape[1] - 1] = 1         #update one-hot vector

    #predicates
        #build column in all predicates
        for pred in self.unaryPredicateMatrices.keys():
            if self.use_sparse:
                # Expand sparse matrix (2, N) -> (2, N+1)
                col = sp.csc_matrix([[0], [1]])
                self.unaryPredicateMatrices[pred] = sp.hstack([self.unaryPredicateMatrices[pred], col]).tocsc()
            else:
                self.unaryPredicateMatrices[pred] = np.insert(self.unaryPredicateMatrices[pred], self.unaryPredicateMatrices[pred].shape[1], 0, 1)
                self.unaryPredicateMatrices[pred][1, self.sizeOfDomain - 1] = 1

        #binary predicates
        for pred in self.binaryPredicateTensors.keys():
            if self.use_sparse:
                t_mat = self.binaryPredicateTensors[pred][0]
                # Expand (N, N) -> (N+1, N+1)
                t_mat = t_mat.tolil()
                t_mat.resize((self.sizeOfDomain, self.sizeOfDomain))
                self.binaryPredicateTensors[pred] = (t_mat.tocsc(), None)
            else:
                #expand tensor
                self.binaryPredicateTensors[pred] = np.insert(self.binaryPredicateTensors[pred], self.binaryPredicateTensors[pred].shape[2], 0, axis=2)
                self.binaryPredicateTensors[pred] = np.insert(self.binaryPredicateTensors[pred], self.binaryPredicateTensors[pred].shape[1], 0, axis=1)
                #update false side of tensor (dim = 1) to be 1 for new element
                self.binaryPredicateTensors[pred][1, self.sizeOfDomain - 1, :] = 1
                self.binaryPredicateTensors[pred][1, :, self.sizeOfDomain - 1] = 1

        #adds element to appropriate predicates in unaryPredicateMatrices and unaryPredicateLookUp
        #and binaryPredicateTensors and binaryPredicateLookUp
        element = tupleToAdd[0]
        if len(tupleToAdd[1]) != 0:
            for item in tupleToAdd[1]:
                if isinstance(item, str):
                    self.updateUnaryPredicate(element, item)
                    self.unaryPredicateLookUp[item][element] = 1.0
                elif isinstance(item, tuple) and len(item) == 2:
                    # (predicate, probability)
                    pred = item[0]
                    prob = item[1]
                    self.updateUnaryPredicate(element, pred, prob)
                    self.unaryPredicateLookUp[pred][element] = float(prob)
                elif isinstance(item, tuple) and len(item) == 3:
                    # item structure: (predicate, otherElement, 'subject'/'object')
                    # 'subject' means new element is subject
                    # 'object' means new element is object
                    pred = item[0]
                    other = item[1]
                    role = item[2]

                    if role == 'subject':
                        pair = (element, other)
                    elif role == 'object':
                        pair = (other, element)
                    else:
                        continue

                    self.updateBinaryPredicate(pair, pred)
                    self.binaryPredicateLookUp[pred].append(pair)


    #remove from domain
    def removeFromDomain(self, element):
        if element not in self.elementLookUp:
            return

        #get index of element to remove
        idx = self.elementLookUp[element]

        #remove from elements list
        self.elements.remove(element)
        #update size of domain
        self.sizeOfDomain -= 1

        #update lookup dictionary
        del self.elementLookUp[element]
        for elem in self.elementLookUp:
            if self.elementLookUp[elem] > idx:
                self.elementLookUp[elem] -= 1

        #update domain matrix
        self.domainMatrix = np.delete(self.domainMatrix, idx, axis=0)
        self.domainMatrix = np.delete(self.domainMatrix, idx, axis=1)

        #update unary predicates
        for pred in self.unaryPredicateMatrices:
            if self.use_sparse:
                mat = self.unaryPredicateMatrices[pred]
                keep_indices = list(range(mat.shape[1]))
                keep_indices.remove(idx)
                self.unaryPredicateMatrices[pred] = mat[:, keep_indices]
            else:
                self.unaryPredicateMatrices[pred] = np.delete(self.unaryPredicateMatrices[pred], idx, axis=1)

        for pred in self.unaryPredicateLookUp:
            if element in self.unaryPredicateLookUp[pred]:
                del self.unaryPredicateLookUp[pred][element]

        #update binary predicates
        for pred in self.binaryPredicateTensors:
            if self.use_sparse:
                t_mat = self.binaryPredicateTensors[pred][0]
                keep = [i for i in range(t_mat.shape[0]) if i != idx]
                t_mat = t_mat[keep, :][:, keep]
                self.binaryPredicateTensors[pred] = (t_mat, None)
            else:
                self.binaryPredicateTensors[pred] = np.delete(self.binaryPredicateTensors[pred], idx, axis=1)
                self.binaryPredicateTensors[pred] = np.delete(self.binaryPredicateTensors[pred], idx, axis=2)

        for pred in self.binaryPredicateLookUp:
            self.binaryPredicateLookUp[pred] = [pair for pair in self.binaryPredicateLookUp[pred] if element not in pair]


    #add unary predicate
    def addUnaryPredicate(self, predicate, listOfElements):
        # normalize elements
        elementsDict = {}
        if isinstance(listOfElements, list):
            for item in listOfElements:
                if isinstance(item, tuple) and len(item) == 2 and (isinstance(item[1], float) or isinstance(item[1], int)):
                    elementsDict[item[0]] = float(item[1])
                else:
                    elementsDict[item] = 1.0
        elif isinstance(listOfElements, dict):
            elementsDict = listOfElements.copy()

        if self.use_sparse:
            predMatrix = sp.lil_matrix((2, self.sizeOfDomain))
            # Populate false values (row 1)
            predMatrix[1, :] = 1

            for elem, prob in elementsDict.items():
                if elem in self.elementLookUp:
                    predMatrix[:,self.elementLookUp[elem]] = np.array([prob, 1 - prob]).reshape(2, 1)

            self.unaryPredicateMatrices[predicate] = predMatrix.tocsc()
        else:
            #build predicate matrix
            predMatrix = np.zeros((2, self.sizeOfDomain))
            predMatrix[1, :] = 1
            for elem, prob in elementsDict.items():
                if elem in self.elementLookUp:
                    predMatrix[:,self.elementLookUp[elem]] = np.array([prob, 1 - prob])
            #add matrix
            self.unaryPredicateMatrices[predicate] = predMatrix

        #add to lookup
        self.unaryPredicateLookUp[predicate] = elementsDict


    #add binary predicate
    def addBinaryPredicate(self, predicate, listOfTuples):
        if self.use_sparse:
            t_mat = sp.lil_matrix((self.sizeOfDomain, self.sizeOfDomain))

            for pair in listOfTuples:
                if pair[0] in self.elementLookUp and pair[1] in self.elementLookUp:
                     t_mat[self.elementLookUp[pair[1]], self.elementLookUp[pair[0]]] = 1.0

            self.binaryPredicateTensors[predicate] = (t_mat.tocsc(), None)
        else:
            #build predicate tensor
            predTensor = np.zeros((2, self.sizeOfDomain, self.sizeOfDomain))
            #get cartesian product
            cartProd = list(itertools.product(*[self.elements for i in [1,2]]))
            for pair in cartProd:
                if pair in listOfTuples:    #if the predicate applies to the ordered pair
                    #fill in true side of tensor (dim = 0) with a 1
                    #[0][obj][subj] = 1
                    predTensor[0][self.elementLookUp[pair[1]]][self.elementLookUp[pair[0]]] = 1
                    #false side of tensor (dim =1 ) will remain a 0
                else:                                           #if the predicate doesn't apply to ordered pair
                    #true size of tensor (dim = 0) will remain a 0
                    #fill in false side of tensor (dim = 1) with a 1
                    predTensor[1][self.elementLookUp[pair[1]]][self.elementLookUp[pair[0]]] = 1
            #add tensor
            self.binaryPredicateTensors[predicate] = predTensor

        #add to lookup
        self.binaryPredicateLookUp[predicate] = listOfTuples


    #add an element to predicate matrix
        #prob = probability that element IS predicate
    def updateUnaryPredicate(self, element, predicate, prob=1):
        if prob == 0:
            self.removeUnaryPredicate(element, predicate)
        else:
            if self.use_sparse:
                # updating sparse matrix
                # Ensure it's in a writable format (lil or dok)
                mat = self.unaryPredicateMatrices[predicate].tolil()
                mat[:, self.elementLookUp[element]] = np.array([prob, 1 - prob]).reshape(2, 1)
                self.unaryPredicateMatrices[predicate] = mat.tocsc()
            else:
                self.unaryPredicateMatrices[predicate][:,self.elementLookUp[element]] = np.array([prob, 1 - prob])


    #add an element to predicate tensor
        #prob = probability that element IS predicate
    def updateBinaryPredicate(self, pair, predicate, prob=1):
        if prob == 0:
            self.removeBinaryPredicate(pair, predicate)
        else:
            if self.use_sparse:
                t_mat = self.binaryPredicateTensors[predicate][0].tolil()
                # t_mat[obj, subj] = prob
                t_mat[self.elementLookUp[pair[1]], self.elementLookUp[pair[0]]] = prob
                self.binaryPredicateTensors[predicate] = (t_mat.tocsc(), None)
            else:
                self.binaryPredicateTensors[predicate][0][self.elementLookUp[pair[1]]][self.elementLookUp[pair[0]]] = prob
                self.binaryPredicateTensors[predicate][1][self.elementLookUp[pair[1]]][self.elementLookUp[pair[0]]] = 1 - prob



    #remove an element from unary predicate matrix
    def removeUnaryPredicate(self, element, predicate):
        if self.use_sparse:
            mat = self.unaryPredicateMatrices[predicate].tolil()
            # Set to False ([0, 1])
            mat[:, self.elementLookUp[element]] = self.isFalse # shape (2,1)
            self.unaryPredicateMatrices[predicate] = mat.tocsc()
        else:
            #update in matrix
            self.unaryPredicateMatrices[predicate][:,self.elementLookUp[element]] = self.isFalse.T

        #update in lookup
        if element in self.unaryPredicateLookUp[predicate]:
            del self.unaryPredicateLookUp[predicate][element]


    def removeBinaryPredicate(self, pair, predicate):
        if self.use_sparse:
            t_mat = self.binaryPredicateTensors[predicate][0].tolil()
            # Set True prob to 0
            t_mat[self.elementLookUp[pair[1]], self.elementLookUp[pair[0]]] = 0
            self.binaryPredicateTensors[predicate] = (t_mat.tocsc(), None)
        else:
            #build temporary tensor
            updatedTensor = self.binaryPredicateTensors[predicate]

            #update temporary tensor
            updatedTensor[0][self.elementLookUp[pair[1]]][self.elementLookUp[pair[0]]] = 0
            updatedTensor[1][self.elementLookUp[pair[1]]][self.elementLookUp[pair[0]]] = 1

            #reassign as permanent tensor
            self.binaryPredicateTensors[predicate] = updatedTensor

        #update in lookup
        if pair in self.binaryPredicateLookUp[predicate]:
            self.binaryPredicateLookUp[predicate].remove(pair)

######################################################

#accessing the items in the world

    #get one hot vector
    def getOneHot(self, element):
        return self.domainMatrix[:,self.elementLookUp[element]].reshape(self.sizeOfDomain, 1)

    #get a predicate
    def getUnaryPredicate(self, predicate):
        return self.unaryPredicateMatrices[predicate]

    def getBinaryPredicate(self, predicate):
        return self.binaryPredicateTensors[predicate]

######################################################

#determining truth
    def unaryOp(self, predicate, element):
        predMat = self.getUnaryPredicate(predicate)
        oneHot = self.getOneHot(element)
        if self.use_sparse:
            # Sparse dot product
            # predMat is (2, N) sparse. oneHot is (N, 1) dense.
            res = predMat.dot(oneHot)
            if isinstance(res, sp.spmatrix):
                res = res.toarray()
            return res
        else:
            return np.tensordot (
                                predMat,
                                oneHot,
                            axes=1)

    def binaryOp(self, predicate, subjElement, objElement):
        subjOneHot = self.getOneHot(subjElement)
        objOneHot = self.getOneHot(objElement)

        if self.use_sparse:
            # Sparse calculation
            # pred is (TrueMatrix, None)
            t_mat, _ = self.getBinaryPredicate(predicate)

            # TrueMatrix is (N, N). Rows=Object, Cols=Subject.
            # P(True) = objOneHot.T * (TrueMatrix * subjOneHot)

            # TrueMatrix * subjOneHot -> vector of shape (N, 1) representing "Objects related to subj".
            vec = t_mat.dot(subjOneHot)

            # objOneHot.T * vec -> scalar (P(True))
            # oneHot is (N, 1). .T is (1, N).

            p_true = objOneHot.T.dot(vec)

            if isinstance(p_true, sp.spmatrix):
                p_true = p_true.toarray()[0,0]
            elif isinstance(p_true, np.ndarray):
                p_true = p_true.item()

            p_false = 1.0 - p_true

            return np.array([[p_true], [p_false]])

        else:
            return np.tensordot (
                                    np.tensordot    (
                                                        self.getBinaryPredicate(predicate),
                                                        subjElement if isinstance(subjElement, np.ndarray) else self.getOneHot(subjElement),
                                                    axes=1).reshape((2, len(self.elementLookUp.keys()))),      #reshape to 2,sizeOfDomain
                                    objElement if isinstance(objElement, np.ndarray) else self.getOneHot(objElement),
                                axes=1)

    def negOp(self, truthValue):
        return np.tensordot (
                                self.negConnect,
                                truthValue,
                            axes=1).reshape((2,1))

    def andOp(self, truthValue1, truthValue2):
        return np.tensordot (
                                np.tensordot    (
                                                    self.andConnect,
                                                    truthValue1,
                                                axes=1).reshape((2,2)).T,
                                truthValue2,
                            axes=1)

    def orOp(self, truthValue1, truthValue2):
        return np.tensordot (
                                np.tensordot    (
                                                    self.orConnect,
                                                    truthValue1,
                                                axes=1).reshape((2,2)).T,
                                truthValue2,
                            axes=1)

    def conditionalOp(self, truthValue1, truthValue2):
        return np.tensordot (
                                np.tensordot    (
                                                    self.conditionalConnect,
                                                    truthValue2,                #the consequent
                                                axes=1).reshape((2,2)).T,
                                truthValue1,                                    #the antecedent
                            axes=1)
