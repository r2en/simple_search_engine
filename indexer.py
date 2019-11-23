import numpy
from typing import List, Set, Dict, Tuple, TypeVar, Callable

from word_embedding.count_vector import CountVector
from word_embedding.tfidf_vector import TfidfVector
from morphological_analysis.mecab_morphological_analysis import MecabMorphologicalAnalysis

def indexer(sentence: List[str], vectorizer: str='count')-> Tuple[numpy.ndarray, Callable]:
    '''
    カタログデータをベクトル変換する
    '''
    corpus = MecabMorphologicalAnalysis().tokenize_docs(sentence)
    if vectorizer == 'count':
        bow = CountVector()
    elif vectorizer == 'tfidf':
        bow = TfidfVector()
    else:
        bow = CountVector() 
    index = bow.fit_transform(corpus)
    return index, bow