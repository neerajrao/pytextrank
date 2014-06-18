from nltk.tokenize import sent_tokenize
from os import path
from pprint import pprint

SRCDIR = path.dirname(path.realpath(__file__))
CORPUSPATH = path.join(SRCDIR,'corpus') # TODO: replace 'corpus' by '../corpus' once directory structure has been finalized

def loadText():
    contents = open(path.join(CORPUSPATH,'test.txt'),'r').read().lower()
    return sent_tokenize(contents.strip())

if __name__ == '__main__':
    sentenceList = loadText()
