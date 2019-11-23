import json
import pandas
import responder
from indexer import indexer
from query_engine import query_engine
from query_pre_processor import query_pre_processor
from query_post_processor import query_post_processor
from product_information_management import product_information_management
from typing import List, Set, Dict, Tuple, TypeVar, Callable

# curl -m '600' -H 'Content-Type: application/json' -X POST -d @resource/query.json http://127.0.0.1:5000/


def batch_process():
    catalog_data = product_information_management()
    index, bow = indexer(catalog_data)
    return index, bow

api = responder.API()

index, bow = batch_process()

@api.route('/')
async def search_engine(req, resp):
    parameter = await req.media()

    sentence = parameter["query"]
    query = query_pre_processor(sentence, bow)

    similarity_distance = query_engine(query, index)
    result = query_post_processor(similarity_distance)

    resp.headers = {"Content-Type": "application/json; charset=utf-8"}
    resp.content = json.dumps(result[0][0], ensure_ascii=False)

if __name__ == '__main__':
    api.run(address='0.0.0.0', port=5000)