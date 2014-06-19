import env
from unittest import TestCase, main
from src import graph_summ
from os import path

SRCDIR = path.dirname(path.realpath(__file__))
CORPUSPATH = path.join(SRCDIR,'../corpus')

class GraphSummTest(TestCase):
    def setUp(self):
        self.file_name = path.join(CORPUSPATH,'test2.txt')
        self.test_text = open(self.file_name,'r').read()
        self.expected_output_pagerank = [u"A major economic great power , Japan has the world 's third-largest economy by nominal GDP and the world 's fourth-largest economy by purchasing power parity .",
                           u"Although Japan has officially renounced its right to declare war , it maintains a modern military with the world 's eighth largest military budget , used for self-defense and peacekeeping roles .",
                           u"Japan has the world 's tenth-largest population , with over 126 million people .",
                           u"The four largest islands are Honshu , Hokkaido , Kyushu , and Shikoku , which together comprise about ninety-seven percent of Japan 's land area ."]
        self.expected_output_hits_auth = [u"A major economic great power , Japan has the world 's third-largest economy by nominal GDP and the world 's fourth-largest economy by purchasing power parity .",
 u"Japan has the world 's tenth-largest population , with over 126 million people .",
 u"Although Japan has officially renounced its right to declare war , it maintains a modern military with the world 's eighth largest military budget , used for self-defense and peacekeeping roles .",
 u"The four largest islands are Honshu , Hokkaido , Kyushu , and Shikoku , which together comprise about ninety-seven percent of Japan 's land area ."]
        self.expected_output_hits_hubs = [u'Japan entered into a long period of isolation in the early 17th century , which was only ended in 1853 when a United States fleet pressured Japan to open to the West .',
 u'However , Japan is also substantially prone to earthquakes and tsunami , having the highest natural disaster risk in the developed world .',
 u'Located in the Pacific Ocean , it lies to the east of the Sea of Japan , China , North Korea , South Korea and Russia , stretching from the Sea of Okhotsk in the north to the East China Sea and Taiwan in the south .',
 u"Honsh\u016b 's Greater Tokyo Area , which includes the de facto capital of Tokyo and several surrounding prefectures , is the largest metropolitan area in the world , with over 30 million residents ."]

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
