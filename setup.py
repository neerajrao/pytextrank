from setuptools import setup

setup(name='graph-ranked-summarization',
      version='0.1',
      description='Text Summarization using Graph-based Ranking Models',
      url='https://github.com/neeraj2608/graph-ranked-summarization',
      author='Raj Rao',
      test_suite='tests',
      install_requires=[
          'numpy',
          'MBSP'
      ],
      dependency_links=['git+https://github.com/clips/MBSP#egg=MBSP'],
      zip_safe=False)
