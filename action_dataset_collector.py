from dataset_collector import get_times_of_spectr_maxima, cut_audio
import os
import constants as const
from elasticsearch import Elasticsearch
import sys

def get_audio_categories(dataset_name):
    es = Elasticsearch(
            [const.ELASTIC_HOST],
            timeout=const.ELASTIC_TIMEOUT
            )
    with open(os.path.join(const.CATEGORIES_PATH, 'action.txt'), 'r') as f:
        for subcategory in f:
            q = subcategory.strip()
            if not q:
                continue
            matches = es.search(index=dataset_name,
                    body={
                        'query': {
                            'match': {
                                'text': q
                                }
                            }
                        }
                    )
            for item in matches['hits']['hits']:
                cut_audio(item["_index"], item['_id'], item["_source"], 'action', q)

if __name__=="__main__":
    names = ['echo_of_moscow_dataset',
             'sk2_dataset',
             'snailkick_dataset',
             'youtube_test']
    for name in names:
        get_audio_categories(name)