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
SUMMARIZATIONMETHODS = ['pagerank','hits_auths','hits_hubs']

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

def summarizeText(text, n=4, method=SUMMARIZATIONMETHODS[0]):
    ''' Returns a list with n most important strings ranked in decreasing order of page rank. '''

    if method not in SUMMARIZATIONMETHODS:
        raise PageRankNotAvailableException("'method' parameter must be one of the following: %s" % SUMMARIZATIONMETHODS)

    sentenceList = tokenizeIntoSentences(text)
    g = Graph.Graph()
    for index, sentence in enumerate(sentenceList):
        g.addSentence(index, getBoW(sentence))

    if method == SUMMARIZATIONMETHODS[0]:
        pageRank = g.getPageRank()
        rankedSentences = map(lambda x: sentenceList[x], np.argsort(pageRank)[::-1])
        return rankedSentences[:n]
    elif method == SUMMARIZATIONMETHODS[1]:
        auth, hubs = g.getHITS()
        rankedSentences = map(lambda x: sentenceList[x], np.argsort(auth)[::-1])
        return rankedSentences[:n]
    else:
        auth, hubs = g.getHITS()
        rankedSentences = map(lambda x: sentenceList[x], np.argsort(hubs)[::-1])
        return rankedSentences[:n]

def testSummarization():
    text = open(path.join(CORPUSPATH,'test3.txt'),'r').read()

    print '#### PageRank ###'
    pprint(summarizeText(text))
    print
    print '#### HITS Auths ###'
    pprint(summarizeText(text, method='hits_auths'))
    print
    print '#### HITS Hubs ###'
    pprint(summarizeText(text, method='hits_hubs'))
    print

if __name__ == '__main__':
    testSummarization()

