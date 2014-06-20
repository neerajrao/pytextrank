from setuptools import setup, find_packages

setup(name='Extractive summarization using Graph-based Ranking Models (TextRank)',
      version='0.1',
      description='Text Summarization using Graph-based Ranking Models',
      url='https://github.com/neeraj2608/graph-ranked-summarization',
      author='Raj Rao',
      test_suite='tests',
      install_requires=[
          'numpy',
          'nltk'
      ],
      zip_safe=False)
