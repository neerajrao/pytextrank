# Tests for pytextrank

import env
from unittest import TestCase, main
import pytextrank
from os import path

TESTDIR = path.dirname(path.realpath(__file__))
CORPUSPATH = path.join(TESTDIR,'corpus')

class GraphSummTest(TestCase):
    def setUp(self):
        self.file_name = path.join(CORPUSPATH,'test2.txt')
        self.test_text = open(self.file_name,'r').read().decode('utf-8','ignore')
        self.expected_output_pagerank  = u"A major economic great power, Japan has the world's third-largest economy by nominal GDP and the world's fourth-largest economy by purchasing power parity. In the late 19th and early 20th centuries, victories in the First Sino-Japanese War, the Russo-Japanese War and World War I allowed Japan to expand its empire during a period of increasing militarism. Japan entered into a long period of isolation in the early 17th century, which was only ended in 1853 when a United States fleet pressured Japan to open to the West. Although Japan has officially renounced its right to declare war, it maintains a modern military with the world's eighth largest military budget, used for self-defense and peacekeeping roles."
        self.expected_output_pagerank_list  = [u"A major economic great power, Japan has the world's third-largest economy by nominal GDP and the world's fourth-largest economy by purchasing power parity.", u'In the late 19th and early 20th centuries, victories in the First Sino-Japanese War, the Russo-Japanese War and World War I allowed Japan to expand its empire during a period of increasing militarism.', u'Japan entered into a long period of isolation in the early 17th century, which was only ended in 1853 when a United States fleet pressured Japan to open to the West.', u"Although Japan has officially renounced its right to declare war, it maintains a modern military with the world's eighth largest military budget, used for self-defense and peacekeeping roles."]
        self.expected_output_hits_auth = u"A major economic great power, Japan has the world's third-largest economy by nominal GDP and the world's fourth-largest economy by purchasing power parity. In the late 19th and early 20th centuries, victories in the First Sino-Japanese War, the Russo-Japanese War and World War I allowed Japan to expand its empire during a period of increasing militarism. Although Japan has officially renounced its right to declare war, it maintains a modern military with the world's eighth largest military budget, used for self-defense and peacekeeping roles. Japan entered into a long period of isolation in the early 17th century, which was only ended in 1853 when a United States fleet pressured Japan to open to the West."
        self.expected_output_hits_hubs = u"In the late 19th and early 20th centuries, victories in the First Sino-Japanese War, the Russo-Japanese War and World War I allowed Japan to expand its empire during a period of increasing militarism. Although Japan has officially renounced its right to declare war, it maintains a modern military with the world's eighth largest military budget, used for self-defense and peacekeeping roles. A major economic great power, Japan has the world's third-largest economy by nominal GDP and the world's fourth-largest economy by purchasing power parity. Japan entered into a long period of isolation in the early 17th century, which was only ended in 1853 when a United States fleet pressured Japan to open to the West."

    def test_summarize_file_pagerank(self):
        self.assertEqual(pytextrank.summarize_file(self.file_name, join=True), self.expected_output_pagerank)

    def test_summarize_text_pagerank(self):
        self.assertEqual(pytextrank.summarize_text(self.test_text, join=True), self.expected_output_pagerank)

    def test_summarize_file_pagerank_list(self):
        self.assertEqual(pytextrank.summarize_file(self.file_name), self.expected_output_pagerank_list)

    def test_summarize_text_pagerank_list(self):
        self.assertEqual(pytextrank.summarize_text(self.test_text), self.expected_output_pagerank_list)

    def test_summarize_file_hits_auth(self):
        self.assertEqual(pytextrank.summarize_file(self.file_name, method='hits_auths', join=True), self.expected_output_hits_auth)

    def test_summarize_text_hits_auth(self):
        self.assertEqual(pytextrank.summarize_text(self.test_text, method='hits_auths', join=True), self.expected_output_hits_auth)

    def test_summarize_file_hits_hubs(self):
        self.assertEqual(pytextrank.summarize_file(self.file_name, method='hits_hubs', join=True), self.expected_output_hits_hubs)

    def test_summarize_text_hits_hubs(self):
        self.assertEqual(pytextrank.summarize_text(self.test_text, method='hits_hubs', join=True), self.expected_output_hits_hubs)

if __name__ == '__main__':
    main()
