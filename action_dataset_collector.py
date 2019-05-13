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
    names = []
    for i in range(1, 13):
        names.append('heads_and_tails_' + str(i))
    names.append('audiobooks_dataset')
    names.append('ht_old_dataset')
    names.append('stories_dataset')
    names.append('interviews_dataset')
    names.append('svoim_hodom_dataset')
    names.append('dud_dataset')

    for name in names:
        get_audio_categories(name)
