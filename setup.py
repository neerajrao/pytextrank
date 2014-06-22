from setuptools import setup, find_packages

setup(name='pytextrank',
      version='0.1',
      description='Text Summarization using Graph-based Ranking Models',
      url='https://github.com/neeraj2608/graph-ranked-summarization',
      author='Raj Rao',
      test_suite='tests',
      install_requires=[
          'numpy',
          'nltk',
          'goose-extractor'
      ],
      zip_safe=False)
