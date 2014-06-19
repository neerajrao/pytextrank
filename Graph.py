# Graph
# Implements Weighted-PageRank

import numpy as np
import re
from math import log
from os import path

DAMPING = 0.85
MAX_ITER = 1000
TOL = 1e-6
PAGERANKMETHODS = ['powermethod','iterativemethod']
SRCDIR = path.dirname(path.realpath(__file__))

class Graph:
    def __init__(self):
        self.vertSet = {}
        self.vertNum = {}
        self.N = 0

    def addVertex(self, key):
        if not key in self.vertSet:
            self.vertSet[key] = Vertex(key)
            self.vertNum[key] = self.N
            self.N += 1

    def addEdge(self, fromKey, toKey, edgeWeight=0):
        if not fromKey in self:
            self.addVertex(fromKey)
        self.getVertex(fromKey).outSet[toKey] = edgeWeight # Adds outcoming connection from vertex
        if not toKey in self:
            self.addVertex(toKey)
        self.getVertex(toKey).inSet[fromKey] = edgeWeight # Adds incoming connection to vertex

    def addSentence(self, index, bow):
        self.addVertex(index)
        self.getVertex(index).bag = bow

    def computeSimilarity(self):
        for firstVertex in self:
            for secondVertex in self:
                edgeWeight = self.calculateSimilarity(secondVertex.bag, firstVertex.bag)
                self.addEdge(firstVertex.id, secondVertex.id, edgeWeight)

    def calculateSimilarity(self, bag1, bag2):
        return float(len(bag1.intersection(bag2)))/(log(len(bag1) + len(bag2))) # normalize by sentence lengths
                                                                                # to avoid bias towards longer sentences

    def getVertex(self, key):
        return self.vertSet.get(key)

    def getVertices(self):
        return self.vertSet.values()

    def getPageRank(self, weighted=True, method=PAGERANKMETHODS[0]):
        '''
        Calculate page rank of graph vertices.

        Arguments
        ---------
        weighted    If True, returns edge-weighted page rank
                    else,    returns normal page rank
        method      'powermethod' uses the power method for calculating the page rank
                    'iterative'   uses the iterative method for calculating the page rank

        Returns
        -------
        pRank       pagerank array of shape [1,N] where N = number of vertices in graph.
        '''
        if method not in PAGERANKMETHODS:
            raise PageRankNotAvailableException("'method' parameter must be one of the following: %s" % PAGERANKMETHODS)

        if self.N == 0:
            raise PageRankNotAvailableException("empty graph!")

        self.computeSimilarity()

        if weighted:
            pRank = np.ones(self.N) / self.N # TODO: change this??
        else:
            pRank = np.ones(self.N) / self.N # initially all 1/N

        if(method == PAGERANKMETHODS[0]): # power method
            if weighted:
                M = self.buildWeightedM()
            else:
                M = self.buildM()

            power = self.powerMethod(pRank, M)
            #print power
            return power
        else: # iterative method
            if weighted:
                A = self.buildWeightedA()
            else:
                A = self.buildA()

            it = self.iterative(pRank, A)
            #print it
            return it

    def powerMethod(self, pRank, M):
        '''
        Calculate pagerank using the power method.
        '''
        M_hat = (DAMPING*M) + ((1-DAMPING)/self.N)

        for i in xrange(MAX_ITER):
            newPRank = np.dot(M_hat,pRank)
            err = np.abs(newPRank-pRank).sum()
            if err < self.N*TOL:
                return newPRank/np.linalg.norm(newPRank)
            pRank = newPRank
        raise PageRankNotAvailableException('Pagerank did not terminate within %d iterations' % MAX_ITER)

    def iterative(self, pRank, A):
        '''
        Calculate pagerank using the iterative method.
        '''
        newPRank = np.dot(DAMPING*A,pRank) + ((1-DAMPING)/self.N)
        err = np.abs(newPRank-pRank).sum()
        if err < TOL:
            return newPRank
        it = self.iterative(newPRank, A)
        return it/np.linalg.norm(it)

    def buildM(self):
        '''
        Builds the Google Matrix M.
        This matrix needs to be calculated only once per invocation of the Pagerank algorithm.
        '''

        A = self.buildA()

        # replace zero columns with initial probability 1/N so we have a column stochastic matrix
        # For example:
        # [[ 0.   0.   0.   0. ]                   [[ 0.   0.25   0.25   0. ]
        #  [ 0.5  0.   0.   0. ]   ---becomes-->    [ 0.5  0.25   0.25   0. ]
        #  [ 0.5  0.   0.   1. ]                    [ 0.5  0.25   0.25   1. ]
        #  [ 0.   0.   0.   0. ]]                   [ 0.   0.25   0.25   0. ]]
        sumA = np.sum(A, axis=0)
        nonzeroindices = np.nonzero(sumA)
        sumA[sumA==0] += float(1)/self.N
        sumA[nonzeroindices] = 0
        M = A + np.tile(sumA,(self.N,1))

        return M

    def buildA(self):
        '''
        Returns an out-degree matrix that has elements set to 1/outdegree(x) for all
        elements (x,y) where vertex x connects to vertex y.

        E.g. for a graph with 4 vertices 0, 1, 2, 3 with connections
        0 -> 1
        0 -> 2
        3 -> 2,
        the following matrix is returned:
        [[ 0.   0.   0.   0. ]
         [ 0.5  0.   0.   0. ]
         [ 0.5  0.   0.   1. ]
         [ 0.   0.   0.   0. ]]

        Explanation:
        element (1,0) is non-zero because node 0 is connected to node 1. It is 0.5 because node 0 has outdegree TWO (two outgoing edges)
        element (2,0) is non-zero because node 0 is connected to node 2. It is 0.5 because node 0 has outdegree TWO (two outgoing edges)
        element (2,3) is non-zero because node 3 is connected to node 2. It is 1   because node 1 has outdegree ONE (one outgoing edges)

        WARNING! Note that the direction is from column to row!!! i.e., (1,0) being non-zero indicates a connection FROM 0 TO 1 and NOT from 1 to 0!!

        Note: This matrix needs to be calculated only once per invocation of the Pagerank algorithm.
        '''
        A = np.zeros((self.N, self.N))
        _A = [elem for vertex in self for elem in \
                            # x connects to vertex
                            map(lambda x: [[self.vertNum[vertex.id], self.vertNum[x]], \
                                            self.getVertex(x).getOutDegree()], \
                                vertex.getIncomingKeys())]
        _A = np.dstack(_A)[0]
        connectedVertices = np.dstack(_A[0])[0]
        outDegrees = _A[1]
        A[list(connectedVertices)] = float(1) / outDegrees
        return A

    def buildWeightedM(self):
        '''
        Builds the Google Matrix M.
        This matrix needs to be calculated only once per invocation of the Pagerank algorithm.
        '''

        A = self.buildWeightedA()

        # replace zero columns with initial probability 1/N so we have a column stochastic matrix
        # For example:
        # [[ 0.   0.   0.   0. ]                   [[ 0.   0.25   0.25   0. ]
        #  [ 0.5  0.   0.   0. ]   ---becomes-->    [ 0.5  0.25   0.25   0. ]
        #  [ 0.5  0.   0.   1. ]                    [ 0.5  0.25   0.25   1. ]
        #  [ 0.   0.   0.   0. ]]                   [ 0.   0.25   0.25   0. ]]
        sumA = np.sum(A, axis=0)
        nonzeroindices = np.nonzero(sumA)
        sumA[sumA==0] += float(1)/self.N
        sumA[nonzeroindices] = 0
        M = A + np.tile(sumA,(self.N,1))

        return M

    def buildWeightedA(self):
        '''
        Returns an out-degree matrix that has elements set to weight(edge(x,y))/(sum(weight(edge(x,n)) for all n))
        for each pair (x,y) where vertex x connects to vertex y.

        E.g. for a graph with 4 vertices 0, 1, 2, 3 with connections
        0 -> 1 with weight 1
        0 -> 2 with weight 2
        3 -> 2 with weight 4
        the following matrix is returned:
        [[ 0.    0.   0.   0. ]
         [ 0.33  0.   0.   0. ]
         [ 0.66  0.   0.   1. ]
         [ 0.    0.   0.   0. ]]

        Explanation:
        element (1,0) is non-zero because node 0 is connected to node 1. It is 0.33 = (1/(1+2))
        element (2,0) is non-zero because node 0 is connected to node 2. It is 0.66 = (2/(1+2))
        element (2,3) is non-zero because node 3 is connected to node 2. It is 1    = (4/4)

        WARNING! Note that the direction is from column to row!!! i.e., (1,0) being non-zero indicates a connection FROM 0 TO 1 and NOT from 1 to 0!!

        Note: This matrix needs to be calculated only once per invocation of the Pagerank algorithm.
        '''
        A = np.zeros((self.N, self.N))
        _A = [elem for vertex in self for elem in \
                            # x connects to vertex
                            map(lambda x: [[self.vertNum[vertex.id], self.vertNum[x]], \
                                            float(self.getVertex(x).outSet[vertex.id])/self.getVertex(x).getTotalOutWeight()], \
                                vertex.getIncomingKeys())]
        _A = np.dstack(_A)[0]
        connectedVertices = np.dstack(_A[0])[0]
        outDegrees = _A[1]
        A[list(connectedVertices)] = outDegrees
        return A

    def __getitem__(self, key):
        return self.getVertex(key)

    def __contains__(self, key):
        return key in self.vertSet

    def __iter__(self):
        return iter(self.getVertices())

class Vertex:
    def __init__(self, key):
        self.id = key
        self.outSet = {} # outgoing edges
        self.inSet = {} # incoming edges
        self.bag = {}

    def getIncomingKeys(self):
        '''
        Returns a list of keys of vertices connected to
        this node by incoming edges.
        '''
        return self.inSet.keys()

    def getInDegree(self):
        return len(self.inSet.keys())

    def getOutDegree(self):
        return len(self.outSet.keys())

    def getOutgoingKeys(self):
        '''
        Returns a list of keys of vertices connected to
        this node by outgoing edges.
        '''
        return self.outSet.keys()

    def getTotalOutWeight(self):
        return sum(self.outSet.values())

class PageRankNotAvailableException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def testGraph():
    g = Graph()
    g.addVertex('a')
    g.addEdge('a','d',10)
    g.addVertex('b')
    g.addVertex('c')
    assert g.N == 4
    assert g.getVertex('a') <> None
    assert g.getVertex('e') == None # non-existent key
    assert 'd' in g.getVertex('a').getOutgoingKeys()
    assert 'a' in g.getVertex('d').getIncomingKeys()

    g.addVertex('a')                                   # key 'a' is already in the graph
    assert g.N == 4                                 # nothing should have been added
    assert 'd' in g.getVertex('a').getOutgoingKeys() # the old node should have been left untouched

    g.addEdge('a','b')
    g.addEdge('c','b')
    assert 'b' in g.getVertex('a').getOutgoingKeys()
    assert 'c' in g.getVertex('b').getIncomingKeys()

    g.getPageRank(weighted=False)

    g1 = Graph()
    g1.addVertex('a')
    g1.addVertex('b')
    g1.addVertex('c')
    g1.addEdge('a','b')
    assert 'b' in g1.getVertex('a').getOutgoingKeys()
    assert 'a' in g1.getVertex('b').getIncomingKeys()

    g1.getPageRank(weighted=False, method='iterativemethod')

    g3 = Graph()
    g3.addVertex('a')
    g3.addVertex('b')
    g3.addVertex('c')
    g3.addVertex('d')
    g3.addEdge('a','c',1)
    g3.addEdge('a','b',2)
    assert g3.getVertex('a').getTotalOutWeight() == 3
    g3.addEdge('d','b',4)
    print g3.buildWeightedA()

if __name__ == '__main__':
    testGraph()
