# Language-agnostic extractive summarization using
# ranked_graph-based ranking models.

from os import path
from pprint import pprint
import ranked_graph
import numpy as np
import re
from nltk.data import load as nltk_load

tokenizer = nltk_load('file:english.pickle')

SRCDIR = path.dirname(path.realpath(__file__))
CORPUSPATH = path.join(SRCDIR,'tests/corpus')
SUMMARIZATIONMETHODS = ['pagerank','hits_auths','hits_hubs']

def build_stop_words_set():
    '''
    Build set of stop words to ignore.
    '''

    # source: http://jmlr.org/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
    return set(open(path.join(SRCDIR,'smartstop.txt'), 'r').read().splitlines())

SMARTSTOPWORDS = build_stop_words_set()

def tokenize_into_sentences(text):
    ''' Tokenizes input text into sentences using NLTK.  '''
    return [s for sentence in map(lambda x: tokenizer.tokenize(unicode(x)), filter(lambda x: len(x) <> 0, text.split('\n'))) for s in sentence]

def get_bow(sentence):
    ''' Returns a bag of words for the sentence '''
    sentenceWords = re.findall(r"[\w']+", sentence)
    cleanWords = []
    for word in sentenceWords:
        if word not in SMARTSTOPWORDS:
            cleanWords.append(word)
    return set(cleanWords)

def summarize_text(text, n=4, method=SUMMARIZATIONMETHODS[0]):
    '''
    Returns a string with n most important sentences in decreasing order of page rank.
    Assumes text is unicode.
    '''

    if method not in SUMMARIZATIONMETHODS:
        raise PageRankNotAvailableException("'method' parameter must be one of the following: %s" % SUMMARIZATIONMETHODS)

    sentenceList = tokenize_into_sentences(text)
    g = ranked_graph.Graph()
    for index, sentence in enumerate(sentenceList):
        g.add_sentence(index, get_bow(sentence))

    if method == SUMMARIZATIONMETHODS[0]:
        pageRank = g.get_pagerank()
        ranked_sentences = map(lambda x: sentenceList[x], np.argsort(pageRank)[::-1])
        return ' '.join(ranked_sentences[:n])
    else:
        auth, hubs = g.get_HITS()
        if method == SUMMARIZATIONMETHODS[1]:
            ranked_sentences = map(lambda x: sentenceList[x], np.argsort(auth)[::-1])
            return ' '.join(ranked_sentences[:n])
        else:
            ranked_sentences = map(lambda x: sentenceList[x], np.argsort(hubs)[::-1])
            return ' '.join(ranked_sentences[:n])

def summarize_file(file_name, n=4, method=SUMMARIZATIONMETHODS[0]):
    text = open(file_name, 'r').read().decode('utf-8','ignore')
    return summarize_text(text, n, method)

def test_summarization():
    text = open(path.join(CORPUSPATH,'test2.txt'),'r').read().decode('utf-8','ignore')

    print '#### PageRank ###'
    print summarize_text(text)
    print
    print '#### HITS Auths ###'
    print summarize_text(text, method='hits_auths')
    print
    print '#### HITS Hubs ###'
    print summarize_text(text, method='hits_hubs')
    print

if __name__ == '__main__':
    test_summarization()

