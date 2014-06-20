# Graph
# Implements Weighted-PageRank
# Implements HITS

import numpy as np
import re
from math import log
from os import path

DAMPING = 0.85
MAX_ITER = 1000
TOL = 1e-6
PAGERANKMETHODS = ['power_method','iterativemethod']

class Graph:
    def __init__(self):
        self.vert_set = {}
        self.vert_num = {}
        self.N = 0

    def add_vertex(self, key):
        if not key in self.vert_set:
            self.vert_set[key] = Vertex(key)
            self.vert_num[key] = self.N
            self.N += 1

    def add_edge(self, from_key, to_key, edge_weight=0):
        if not from_key in self:
            self.add_vertex(from_key)
        self.get_vertex(from_key).out_set[to_key] = edge_weight # Adds outcoming connection from vertex
        if not to_key in self:
            self.add_vertex(to_key)
        self.get_vertex(to_key).in_set[from_key] = edge_weight # Adds incoming connection to vertex

    def add_sentence(self, index, bow):
        self.add_vertex(index)
        self.get_vertex(index).bag = bow

    def compute_similarity(self):
        for first_vertex in self:
            for second_vertex in self:
                edge_weight = self.calculate_similarity(second_vertex.bag, first_vertex.bag)
                if edge_weight > 0:
                    self.add_edge(first_vertex.id, second_vertex.id, edge_weight)

    def calculate_similarity(self, bag1, bag2):
        if (not bag1) or (not bag2):
            return 0
        else:
            return float(len(bag1.intersection(bag2)))/(log(len(bag1) + len(bag2))) # normalize by sentence lengths
                                                                                    # to avoid bias towards longer sentences

    def get_vertex(self, key):
        return self.vert_set.get(key)

    def get_vertices(self):
        return self.vert_set.values()

    def get_pagerank(self, weighted=True, method=PAGERANKMETHODS[0]):
        '''
        Calculate page rank of graph vertices.

        Arguments
        ---------
        weighted    If True, returns edge-weighted page rank
                    else,    returns normal page rank
        method      'power_method' uses the power method for calculating the page rank
                    'iterative'   uses the iterative method for calculating the page rank

        Returns
        -------
        pagerank       pagerank array of shape [1,N] where N = number of vertices in graph.
        '''
        if method not in PAGERANKMETHODS:
            raise PageRankNotAvailableException("'method' parameter must be one of the following: %s" % PAGERANKMETHODS)

        if self.N == 0:
            raise PageRankNotAvailableException("empty graph!")

        self.compute_similarity()

        if weighted:
            pagerank = np.ones(self.N) / self.N # initially all 1/N
        else:
            pagerank = np.ones(self.N) / self.N # initially all 1/N

        if(method == PAGERANKMETHODS[0]): # power method
            if weighted:
                M = self.build_weighted_M()
            else:
                M = self.build_M()

            power = self.power_method(pagerank, M)
            #print power
            return power
        else: # iterative method
            if weighted:
                A = self.build_weighted_A()
            else:
                A = self.build_A()

            it = self.iterative(pagerank, A)
            #print it
            return it

    def power_method(self, pagerank, M):
        '''
        Calculate pagerank using the power method.
        '''
        M_hat = (DAMPING*M) + ((1-DAMPING)/self.N)

        for i in xrange(MAX_ITER):
            new_pagerank = np.dot(M_hat,pagerank)
            err = np.abs(new_pagerank-pagerank).sum()
            if err < self.N*TOL:
                return new_pagerank/np.linalg.norm(new_pagerank)
            pagerank = new_pagerank
        raise PageRankNotAvailableException('Pagerank did not terminate within %d iterations' % MAX_ITER)

    def iterative(self, pagerank, A):
        '''
        Calculate pagerank using the iterative method.
        '''
        new_pagerank = np.dot(DAMPING*A,pagerank) + ((1-DAMPING)/self.N)
        err = np.abs(new_pagerank-pagerank).sum()
        if err < TOL:
            return new_pagerank
        it = self.iterative(new_pagerank, A)
        return it/np.linalg.norm(it)

    def build_M(self):
        '''
        Builds the Google Matrix M.
        This matrix needs to be calculated only once per invocation of the Pagerank algorithm.
        '''

        A = self.build_A()

        # replace zero columns with initial probability 1/N so we have a column stochastic matrix
        # For example:
        # [[ 0.   0.   0.   0. ]                   [[ 0.   0.25   0.25   0. ]
        #  [ 0.5  0.   0.   0. ]   ---becomes-->    [ 0.5  0.25   0.25   0. ]
        #  [ 0.5  0.   0.   1. ]                    [ 0.5  0.25   0.25   1. ]
        #  [ 0.   0.   0.   0. ]]                   [ 0.   0.25   0.25   0. ]]
        sumA = np.sum(A, axis=0)
        nonzero_indices = np.nonzero(sumA)
        sumA[sumA==0] += float(1)/self.N
        sumA[nonzero_indices] = 0
        M = A + np.tile(sumA,(self.N,1))

        return M

    def build_A(self):
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
        _A = [elem for vertex in self for elem in
                            map(lambda x: [[self.vert_num[vertex.id], self.vert_num[x]], # x connects to vertex
                                            self.get_vertex(x).get_out_degree()],
                                vertex.get_incoming_keys())]
        connected_vertices, out_degrees = np.dstack(_A)[0]
        connected_vertices = np.dstack(connected_vertices)[0]
        A[list(connected_vertices)] = float(1) / out_degrees
        return A

    def build_weighted_M(self):
        '''
        Builds the Google Matrix M.
        This matrix needs to be calculated only once per invocation of the Pagerank algorithm.
        '''

        A = self.build_weighted_A()

        # replace zero columns with initial probability 1/N so we have a column stochastic matrix
        # For example:
        # [[ 0.   0.   0.   0. ]                   [[ 0.   0.25   0.25   0. ]
        #  [ 0.5  0.   0.   0. ]   ---becomes-->    [ 0.5  0.25   0.25   0. ]
        #  [ 0.5  0.   0.   1. ]                    [ 0.5  0.25   0.25   1. ]
        #  [ 0.   0.   0.   0. ]]                   [ 0.   0.25   0.25   0. ]]
        sumA = np.sum(A, axis=0)
        nonzero_indices = np.nonzero(sumA)
        sumA[sumA==0] += float(1)/self.N
        sumA[nonzero_indices] = 0
        M = A + np.tile(sumA,(self.N,1))

        return M

    def build_weighted_A(self):
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
        _A = [elem for vertex in self for elem in
                            map(lambda x: [[self.vert_num[vertex.id], self.vert_num[x]], # x connects to vertex
                                            float(self.get_vertex(x).out_set[vertex.id])/self.get_vertex(x).get_total_out_weight()],
                                vertex.get_incoming_keys())]
        connected_vertices, out_weight = np.dstack(_A)[0]
        connected_vertices = np.dstack(connected_vertices)[0]
        A[list(connected_vertices)] = out_weight
        return A

    def get_HITS(self):
        """ Calculate page rank of graph vertices. """

        self.compute_similarity()

        auths = np.ones(self.N) #/self.N
        hubs = np.ones(self.N) #/self.N

        A = self.build_weighted_A()
        At = A.T

        for i in xrange(MAX_ITER):
            new_auths = np.dot(A, hubs)
            new_hubs = np.dot(At, new_auths)

            new_auths = new_auths/np.sum(new_auths)
            new_hubs = new_hubs/np.sum(new_hubs)

            auths_error = np.abs(new_auths-auths).sum()
            hubs_error = np.abs(new_hubs-hubs).sum()

            #print auths_error, hubs_error

            if (auths_error < self.N*TOL) or (hubs_error < self.N*TOL):
                return new_auths, new_hubs

            auths = new_auths
            hubs = new_hubs

        raise HITSNotAvailableException('HITS did not terminate within %d iterations' % MAX_ITER)

    def __getitem__(self, key):
        return self.get_vertex(key)

    def __contains__(self, key):
        return key in self.vert_set

    def __iter__(self):
        return iter(self.get_vertices())

class Vertex:
    def __init__(self, key):
        self.id = key
        self.out_set = {} # outgoing edges
        self.in_set = {} # incoming edges
        self.bag = set()

    def get_incoming_keys(self):
        '''
        Returns a list of keys of vertices connected to
        this node by incoming edges.
        '''
        return self.in_set.keys()

    def get_in_degree(self):
        return len(self.in_set.keys())

    def get_out_degree(self):
        return len(self.out_set.keys())

    def get_outgoing_keys(self):
        '''
        Returns a list of keys of vertices connected to
        this node by outgoing edges.
        '''
        return self.out_set.keys()

    def get_total_out_weight(self):
        return sum(self.out_set.values())

class PageRankNotAvailableException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class HITSNotAvailableException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def test_graph():
    g = Graph()
    g.add_vertex('a')
    g.add_edge('a','d',10)
    g.add_vertex('b')
    g.add_vertex('c')
    assert g.N == 4
    assert g.get_vertex('a') <> None
    assert g.get_vertex('e') == None # non-existent key
    assert 'd' in g.get_vertex('a').get_outgoing_keys()
    assert 'a' in g.get_vertex('d').get_incoming_keys()

    g.add_vertex('a')                                   # key 'a' is already in the graph
    assert g.N == 4                                 # nothing should have been added
    assert 'd' in g.get_vertex('a').get_outgoing_keys() # the old node should have been left untouched

    g.add_edge('a','b')
    g.add_edge('c','b')
    assert 'b' in g.get_vertex('a').get_outgoing_keys()
    assert 'c' in g.get_vertex('b').get_incoming_keys()

    g.get_pagerank(weighted=False)

    g1 = Graph()
    g1.add_vertex('a')
    g1.add_vertex('b')
    g1.add_vertex('c')
    g1.add_edge('a','b',10)
    assert 'b' in g1.get_vertex('a').get_outgoing_keys()
    assert 'a' in g1.get_vertex('b').get_incoming_keys()

    g1.get_pagerank(weighted=False, method='iterativemethod')
    print g1.get_HITS()

    g3 = Graph()
    g3.add_vertex('a')
    g3.add_vertex('b')
    g3.add_vertex('c')
    g3.add_vertex('d')
    g3.add_edge('a','c',1)
    g3.add_edge('a','b',2)
    assert g3.get_vertex('a').get_total_out_weight() == 3
    g3.add_edge('d','b',4)
    print g3.build_weighted_A()

if __name__ == '__main__':
    test_graph()
