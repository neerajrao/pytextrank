# Tests for graph_summ

import env
from unittest import TestCase, main
import graph_summ
from os import path

TESTDIR = path.dirname(path.realpath(__file__))
CORPUSPATH = path.join(TESTDIR,'corpus')

class GraphSummTest(TestCase):
    def setUp(self):
        self.file_name = path.join(CORPUSPATH,'test2.txt')
        self.test_text = open(self.file_name,'r').read().decode('utf-8','ignore')
        self.expected_output_pagerank  = u"A major economic great power, Japan has the world's third-largest economy by nominal GDP and the world's fourth-largest economy by purchasing power parity. In the late 19th and early 20th centuries, victories in the First Sino-Japanese War, the Russo-Japanese War and World War I allowed Japan to expand its empire during a period of increasing militarism. Japan entered into a long period of isolation in the early 17th century, which was only ended in 1853 when a United States fleet pressured Japan to open to the West. Although Japan has officially renounced its right to declare war, it maintains a modern military with the world's eighth largest military budget, used for self-defense and peacekeeping roles."
        self.expected_output_hits_auth = u"A major economic great power, Japan has the world's third-largest economy by nominal GDP and the world's fourth-largest economy by purchasing power parity. Japan entered into a long period of isolation in the early 17th century, which was only ended in 1853 when a United States fleet pressured Japan to open to the West. Although Japan has officially renounced its right to declare war, it maintains a modern military with the world's eighth largest military budget, used for self-defense and peacekeeping roles. In the late 19th and early 20th centuries, victories in the First Sino-Japanese War, the Russo-Japanese War and World War I allowed Japan to expand its empire during a period of increasing militarism."
        self.expected_output_hits_hubs = u"Japan is a great power and is a member of the United Nations, the G8, the G20. Honsh\u016b's Greater Tokyo Area, which includes the de facto capital of Tokyo and several surrounding prefectures, is the largest metropolitan area in the world, with over 30 million residents. In the late 19th and early 20th centuries, victories in the First Sino-Japanese War, the Russo-Japanese War and World War I allowed Japan to expand its empire during a period of increasing militarism. Nearly two decades of internal conflict and insurrection followed before the Meiji Emperor was restored as head of state in 1868 and the Empire of Japan was proclaimed, with the Emperor as a divine symbol of the nation."

    def test_summarize_file_pagerank(self):
        self.assertEqual(graph_summ.summarize_file(self.file_name), self.expected_output_pagerank)

    def test_summarize_text_pagerank(self):
        self.assertEqual(graph_summ.summarize_text(self.test_text), self.expected_output_pagerank)

    def test_summarize_file_hits_auth(self):
        self.assertEqual(graph_summ.summarize_file(self.file_name, method='hits_auths'), self.expected_output_hits_auth)

    def test_summarize_text_hits_auth(self):
        self.assertEqual(graph_summ.summarize_text(self.test_text, method='hits_auths'), self.expected_output_hits_auth)

    def test_summarize_file_hits_hubs(self):
        self.assertEqual(graph_summ.summarize_file(self.file_name, method='hits_hubs'), self.expected_output_hits_hubs)

    def test_summarize_text_hits_hubs(self):
        self.assertEqual(graph_summ.summarize_text(self.test_text, method='hits_hubs'), self.expected_output_hits_hubs)

if __name__ == '__main__':
    main()
