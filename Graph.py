# Graph

class Graph:
    def __init__(self):
        self.vertset = {}
        self.size = 0

    def addVertex(self, key):
        if not key in self.vertset:
            self.size += 1
            self.vertset[key] = Vertex(key)

    def getVertex(self, key):
        return self.vertset.get(key)

    def __getitem__(self, key):
        return getVertex(key)

    def __contains__(self, key):
        return key in self.vertset

    def getVertices(self):
        return self.vertset.values()

    def __iter__(self):
        return iter(getVertices())

    def addEdge(self, fromKey, toKey, edgeWeight=0):
        if not fromKey in self:
            addVertex(fromKey)
        self.getVertex(fromKey).addNeighbor(toKey, edgeWeight)

class Vertex:
    def __init__(self, key):
        self.key = key
        self.adjset = {}

    def addNeighbor(self, key, weight=0):
        '''
        Adds connection to vertex with key = key
        '''
        self.adjset[key] = weight

    def getConnectedKeys(self):
        return self.adjset.keys()

    def getWeight(self, v):
        return self.adjset[v.key]

def testGraph():
    g = Graph()
    g.addVertex('a')
    g.getVertex('a').addNeighbor('a1', 10)
    g.addVertex('b')
    g.addVertex('c')
    assert g.size == 3
    assert g.getVertex('a') <> None
    assert g.getVertex('d') == None # non-existent key

    g.addVertex('a')                                   # key 'a' is already in the graph
    assert g.size == 3                                 # nothing should be added
    assert 'a1' in g.getVertex('a').getConnectedKeys() # the old node should be left untouched

    g.addEdge('a','b')
    assert 'b' in g.getVertex('a').getConnectedKeys()

if __name__ == '__main__':
    testGraph()
