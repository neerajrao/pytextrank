# Language-agnostic extractive summarization using
# Graph-based ranking models.

from os import path
from pprint import pprint
import Graph
import numpy as np
import re
from MBSP.tokenizer import split

SRCDIR = path.dirname(path.realpath(__file__))
CORPUSPATH = path.join(SRCDIR,'corpus') # TODO: replace 'corpus' by '../corpus' once directory structure has been finalized

def build_stop_words_set():
    '''
    Build set of stop words to ignore.
    '''

    # source: http://jmlr.org/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
    return set(open(path.join(SRCDIR,'smartstop.txt'), 'r').read().splitlines())

smartStopWords = build_stop_words_set()

def tokenizeIntoSentences(text):
    ''' Tokenizes input text into sentences using the MBSP parser.  '''
    return split(text.strip())

def getBoW(sentence):
    ''' Returns a bag of words for the sentence '''
    sentenceWords = re.findall(r"[\w']+", sentence)
    cleanWords = []
    for word in sentenceWords:
        if word not in smartStopWords:
            cleanWords.append(word)
    return set(cleanWords)

def summarizeText(text, n=5):
    ''' Returns a list of length n with most important strings as ranked by page rank. '''
    sentenceList = tokenizeIntoSentences(text)
    g = Graph.Graph()
    for index, sentence in enumerate(sentenceList):
        g.addSentence(index, getBoW(sentence))
    pageRank = g.getPageRank()
    rankedSentences = map(lambda x: sentenceList[x], np.argsort(pageRank)[::-1])
    return rankedSentences[:n]

if __name__ == '__main__':
    text = open(path.join(CORPUSPATH,'test1.txt'),'r').read()
    rankedSentences = summarizeText(text)
    pprint(rankedSentences)
