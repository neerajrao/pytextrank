# Graph

class Graph:
    def __init__():
        self.vertset = {}

class Vertex:
    def __init__(self, key):
        self.key = key
        self.adjset = {}

    def addneighbor(self, v, weight=0):
        self.adjset[v.key] = weight

    def getconnections():
        return self.adjset.keys()

    def getweight(self, v):
        return self.adjset[v.key]

if __name__ == '__main__':
    pass
