# Tests for graph_summ

import env
from unittest import TestCase, main
from ranked_graph import Graph
from os import path
import numpy as np

TESTDIR = path.dirname(path.realpath(__file__))
CORPUSPATH = path.join(TESTDIR, 'corpus')

class RankedGraphTest(TestCase):
    def setUp(self):
        self.g = Graph()

    def test_add_vertex(self):
        self.g.add_vertex('a')
        self.g.add_edge('a','d', 10)
        self.g.add_vertex('b')
        self.g.add_vertex('c')
        self.assertEqual(self.g.N, 4)
        self.assertTrue(self.g.get_vertex('a'))
        self.assertIsNone(self.g.get_vertex('e')) # non-existent key
        self.assertTrue('d' in self.g.get_vertex('a').get_outgoing_keys())
        self.assertTrue('a' in self.g.get_vertex('d').get_incoming_keys())

    def test_re_add_existing_node_does_nothing(self):
        self.g.add_vertex('a')
        self.g.add_edge('a', 'd', 10)
        self.g.add_vertex('a')                                             # key 'a' is already in the graph
        self.assertEqual(self.g.N, 2)                                      # nothing should have been added
        self.assertTrue('d' in self.g.get_vertex('a').get_outgoing_keys()) # the old node should have been left untouched

    def test_re_add_existing_edge_updates_edge(self):
        self.g.add_vertex('a')
        self.g.add_edge('a', 'd', 10)
        self.assertEqual(self.g.get_vertex('a').out_set['d'], 10)
        self.g.add_edge('a', 'd', 3)                                         # should update edge weight
        self.assertEqual(self.g.get_vertex('a').out_set['d'], 3)
        self.g.add_edge('a', 'd')                                            # should update weight to default of 1
        self.assertEqual(self.g.get_vertex('a').out_set['d'], 1)

    def test_get_pagerank(self):
        self.g.add_vertex('a')
        self.g.add_edge('a','d', 10)
        self.g.add_vertex('b')
        self.g.add_vertex('c')
        self.g.add_edge('a', 'b')
        self.g.add_edge('c', 'b')

        self.assertTrue(any(self.g.get_pagerank(weighted=False)))
        self.assertTrue(any(self.g.get_pagerank(weighted=False, method='iterativemethod')))

    def test_get_HITS(self):
        self.g.add_vertex('a')
        self.g.add_edge('a','d', 10)
        self.g.add_vertex('b')
        self.g.add_vertex('c')
        self.g.add_edge('a', 'b')
        self.g.add_edge('c', 'b')

        auths, hubs = self.g.get_HITS()
        self.assertEqual(self.g.get_vertex('a').get_total_out_weight(), 11)
        self.assertTrue(any(auths))
        self.assertTrue(any(hubs))

if __name__ == '__main__':
    main()
