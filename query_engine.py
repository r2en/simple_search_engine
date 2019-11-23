import numpy
from search.cosine_similarity import cosine_similarity
from typing import List, Set, Dict, Tuple, TypeVar, Callable

def query_engine(query_vec: numpy.ndarray, index_vec: numpy.ndarray) -> List:
    '''
    インデックスと、クエリのベクトル距離を計算する
    '''
    similarity_distances = []
    for i in range(len(index_vec)):
        v1 = index_vec[i]
        v2 = query_vec[0]
        similarity_distance = cosine_similarity(v1, v2)
        similarity_distances.append(similarity_distance)
    return similarity_distances