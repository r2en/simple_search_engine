import pandas
from typing import List, Set, Dict, Tuple, TypeVar, Callable

def product_information_management() -> List[str]:
    '''
    カタログデータを取得する
    '''
    sentence = pandas.read_csv('catalog_data/question.csv')['question'].values.tolist()
    return sentence