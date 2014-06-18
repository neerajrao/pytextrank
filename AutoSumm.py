# Language-agnostic extractive summarization using
# Graph-based ranking models.

from nltk.tokenize import sent_tokenize
from os import path
from pprint import pprint
import Graph
import numpy as np
import re

SRCDIR = path.dirname(path.realpath(__file__))
CORPUSPATH = path.join(SRCDIR,'corpus') # TODO: replace 'corpus' by '../corpus' once directory structure has been finalized

def tokenizeIntoSentences(text):
    contents = re.sub('\'s|(\n\n)|(\r\n)|-+|\'\'|["_]', ' ', text) # remove \r\n, apostrophes, quotes and dashes
    return sent_tokenize(contents.strip())

def summarizeText(text, n=5):
    sentenceList = tokenizeIntoSentences(text.lower())
    g = Graph.Graph()
    for index, sentence in enumerate(sentenceList):
        g.addSentence(index, sentence)
    pageRank = g.getPageRank()
    rankedSentences = map(lambda x: sentenceList[x], np.argsort(pageRank)[::-1])
    return rankedSentences[:n]

if __name__ == '__main__':
    text = open(path.join(CORPUSPATH,'test1.txt'),'r').read()
    rankedSentences = summarizeText(text)
    pprint(rankedSentences)
