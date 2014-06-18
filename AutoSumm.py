# Extractive summarization using Graph-based ranking models.

from nltk.tokenize import sent_tokenize
from os import path
from pprint import pprint
import Graph
import numpy as np
import re

SRCDIR = path.dirname(path.realpath(__file__))
CORPUSPATH = path.join(SRCDIR,'corpus') # TODO: replace 'corpus' by '../corpus' once directory structure has been finalized

def loadText():
    text = open(path.join(CORPUSPATH,'test.txt'),'r').read().lower()
    contents = re.sub('\'s|(\n\n)|(\r\n)|-+|\'\'|["_]', ' ', text) # remove \r\n, apostrophes, quotes and dashes
    return sent_tokenize(contents.strip())

if __name__ == '__main__':
    sentenceList = loadText()
    g = Graph.Graph()
    for index, sentence in enumerate(sentenceList):
        g.addSentence(index, sentence)
    pageRank = g.getPlainVanillaPageRank()
    print pageRank
    print np.argsort(pageRank)
    #print sentenceList[np.argsort(pageRank)[::-1]]
