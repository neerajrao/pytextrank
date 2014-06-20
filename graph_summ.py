# Language-agnostic extractive summarization using
# ranked_graph-based ranking models.

from os import path
from pprint import pprint
import ranked_graph
import numpy as np
import re
from goose import Goose
from nltk.data import load as nltk_load

tokenizer = nltk_load('file:english.pickle')

SRCDIR = path.dirname(path.realpath(__file__))
CORPUSPATH = path.join(SRCDIR,'tests/corpus')
SUMMARIZATIONMETHODS = ['pagerank','hits_auths','hits_hubs']

def _build_stop_words_set():
    '''
    Build set of stop words to ignore.
    '''

    # source: http://jmlr.org/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
    return set(open(path.join(SRCDIR,'smartstop.txt'), 'r').read().splitlines())

SMARTSTOPWORDS = _build_stop_words_set()

def _tokenize_into_sentences(text):
    ''' Tokenizes input text into sentences using NLTK.  '''
    return [s for sentence in map(lambda x: tokenizer.tokenize(unicode(x)), filter(lambda x: len(x) <> 0, text.split('\n'))) for s in sentence]

def _get_bow(sentence):
    ''' Returns a bag of words for the sentence '''
    sentenceWords = re.findall(r"[\w']+", sentence)
    cleanWords = []
    for word in sentenceWords:
        if word not in SMARTSTOPWORDS:
            cleanWords.append(word)
    return set(cleanWords)

def summarize_text(text, n=4, method=SUMMARIZATIONMETHODS[0], join=False):
    '''
    Summarize text.
    Assumes text is decoded bytestream.

    Arguments
    ---------
    text:       text to summarize
    n:          number of sentences to return
    method:     summarization algorithm to use. Can be pagerank, HITS authority scores
                or HITS hubs scores
    join:       True: return results as a string
                False (default): return results as a list of string

    Returns
    -------
    n most important sentences from text in decreasing order of page rank. Results are passed
    back as a list if join argument is False, or as a string is join argument is True
    '''

    if method not in SUMMARIZATIONMETHODS:
        raise PageRankNotAvailableException("'method' parameter must be one of the following: %s" % SUMMARIZATIONMETHODS)

    sentenceList = _tokenize_into_sentences(text)
    g = ranked_graph.Graph()
    for index, sentence in enumerate(sentenceList):
        g.add_sentence(index, _get_bow(sentence))

    if method == SUMMARIZATIONMETHODS[0]:
        pageRank = g.get_pagerank()
        ranked_sentences = map(lambda x: sentenceList[x], np.argsort(pageRank)[::-1])
    else:
        auth, hubs = g.get_HITS()
        if method == SUMMARIZATIONMETHODS[1]:
            ranked_sentences = map(lambda x: sentenceList[x], np.argsort(auth)[::-1])
            #return ' '.join(ranked_sentences[:n])
        else:
            ranked_sentences = map(lambda x: sentenceList[x], np.argsort(hubs)[::-1])
            #return ' '.join(ranked_sentences[:n])

    if join:
        return ' '.join(ranked_sentences[:n])
    else:
        return ranked_sentences[:n]

def summarize_file(file_name, n=4, method=SUMMARIZATIONMETHODS[0], join=False):
    ''' Summarize text from file '''
    text = open(file_name, 'r').read().decode('utf-8','ignore')
    return summarize_text(text, n=n, method=method, join=join)

def summarize_url(url, n=4, method=SUMMARIZATIONMETHODS[0], join=False):
    ''' Summarize article from url '''
    # taken from https://github.com/xiaoxu193/PyTeaser
    try:
        extracted_article = _pluck_the_goose(url)
    except IOError:
        print 'Error reading from %s' % url
        return None

    if not extracted_article or not extracted_article.cleaned_text or not extracted_article.title:
        return None

    return summarize_text(extracted_article.cleaned_text, n=n, method=method, join=join)

def _pluck_the_goose(inurl):
    ''' Extracts article from URL using Goose '''
    # taken from https://github.com/xiaoxu193/PyTeaser
    #extract URL information using Python Goose
    try:
        extracted_article = Goose().extract(url=inurl)
        return extracted_article
    except ValueError:
        print 'This goose hath no feathers.'
        return None
    return None

def _test_summarization():
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
    #print '#### PageRank ###'
    #print summarize_url('http://www.washingtonpost.com/world/islamic-militants-bear-down-on-iraqi-forces-seize-chemical-weapons-facility/2014/06/20/b69df9c2-8301-461a-9258-bb1fa1c470eb_story.html?hpid=z1')
    #print

if __name__ == '__main__':
    _test_summarization()

