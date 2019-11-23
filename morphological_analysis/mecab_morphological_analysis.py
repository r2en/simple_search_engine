import MeCab
from gensim.models import KeyedVectors
from typing import List, Set, Dict, Tuple, TypeVar, Callable
from morphological_analysis.base_morphological_analysis import BaseMorphologicalAnalysis

class MecabMorphologicalAnalysis(BaseMorphologicalAnalysis):
    def __init__(self) -> None:
        super().__init__()
        self.tagger = MeCab.Tagger("-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

    def tokenize(self, text: str) -> List:
        return self.tagger.parse(text).strip().split(" ")

    def tokenize_docs(self, docs: List) -> List:
        corpus = [self.tagger.parse(sentence).replace(' \n', '') for sentence in docs]
        return corpus