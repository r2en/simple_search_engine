import numpy
from scipy import spatial
from typing import List, Set, Dict, Tuple, TypeVar, Callable

def cosine_similarity(v1: numpy.ndarray, v2: numpy.ndarray) -> numpy.float64:
    '''
    文書ベクトル間の類似度を測る
    1に近ければ類似、0に近ければ相違
    '''
    return 1 - spatial.distance.cosine(v1, v2)