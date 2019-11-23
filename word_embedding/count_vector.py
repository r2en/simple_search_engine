import numpy
from word_embedding.base_vector import BaseVector
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Set, Dict, Tuple, TypeVar, Callable

class CountVector(BaseVector):
    '''
    テキストをBoWでベクトル変換するためのクラス
    '''
    def __init__(self) -> None:
        super().__init__()
        self.bow = CountVectorizer()
        
    def fit_transform(self, corpus: List) -> numpy.ndarray:
        index_vec = self.bow.fit_transform(corpus).toarray()
        return index_vec
    
    def transform(self, corpus: List) -> numpy.ndarray:
        query_vec = self.bow.transform(corpus).toarray()
        return query_vec