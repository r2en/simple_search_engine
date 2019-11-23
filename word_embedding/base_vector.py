from abc import ABCMeta, abstractmethod
from typing import List, Set, Dict, Tuple, TypeVar, Callable

class BaseVector(metaclass=ABCMeta):
    '''
    テキストをベクトル変換するための親クラス
    '''

    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def fit_transform(self, corpus: List) -> None:
        pass
    
    @abstractmethod
    def transform(self, corpus: List) -> None:
        pass