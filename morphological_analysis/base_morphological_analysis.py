from abc import ABCMeta, abstractmethod
from typing import List, Set, Dict, Tuple, TypeVar, Callable

class BaseMorphologicalAnalysis(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def tokenize(self, docs: str) -> List:
        pass
    
    @abstractmethod
    def tokenize_docs(self, docs: List) -> List:
        pass