import numpy
import pandas
from typing import List, Set, Dict, Tuple, TypeVar, Callable

def query_post_processor(similarity_distance: List[numpy.float64]) -> numpy.ndarray:
    '''
    類似度上位三件のインデックスを取得し
    カタログデータから類似文書を返す
    '''
    rank = numpy.array(similarity_distance).argsort()[-3:][::-1]
    catalog_data = pandas.read_csv('catalog_data/answer.csv')
    return numpy.array(catalog_data)[rank]