import env
from unittest import TestCase, main
from src import GraphSumm
from os import path

SRCDIR = path.dirname(path.realpath(__file__))
CORPUSPATH = path.join(SRCDIR,'../corpus')

class GraphSummTest(TestCase):
    def setUp(self):
        self.file_name = path.join(CORPUSPATH,'test2.txt')
        self.test_text = open(self.file_name,'r').read()
        self.expected_output = [u"A major economic great power , Japan has the world 's third-largest economy by nominal GDP and the world 's fourth-largest economy by purchasing power parity .",
                           u"Although Japan has officially renounced its right to declare war , it maintains a modern military with the world 's eighth largest military budget , used for self-defense and peacekeeping roles .",
                           u"Japan has the world 's tenth-largest population , with over 126 million people .",
                           u"The four largest islands are Honshu , Hokkaido , Kyushu , and Shikoku , which together comprise about ninety-seven percent of Japan 's land area ."]

    def test_summarize_file(self):
        self.assertEqual(GraphSumm.summarize_file(self.file_name), self.expected_output)

    def test_summarize_text(self):
        self.assertEqual(GraphSumm.summarize_text(self.test_text), self.expected_output)

if __name__ == '__main__':
    main()
