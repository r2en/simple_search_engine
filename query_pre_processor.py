import numpy
from typing import List, Set, Dict, Tuple, TypeVar, Callable
from morphological_analysis.mecab_morphological_analysis import MecabMorphologicalAnalysis

def query_pre_processor(query: List, bow: Callable) -> numpy.ndarray:
    '''
    クエリをベクトル変換する
    '''
    corpus = MecabMorphologicalAnalysis().tokenize_docs(query)
    vec = bow.transform(corpus)
    return vec