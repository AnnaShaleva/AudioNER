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
    with open(os.path.join(const.CATEGORIES_PATH, 'confirm.txt'), 'r') as f:
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
                cut_audio(item["_index"], item['_id'], item["_source"], 'confirm', q)

if __name__=="__main__":
    #name = 'audiobooks_dataset'
    name = 'stories_dataset'
    get_audio_categories(name)
    #name = 'ht_old_dataset'
    #get_audio_categories(name)
    #for i in range(1, 13):
    #    name = 'heads_and_tails_' + str(i)
    #    get_audio_categories(name)
    #name = './data/echo_of_moscow_dataset/categories_samples/'
    #get_audio_categories(name)
    #name = './data/snailkick_dataset/categories_samples/'
    #get_audio_categories(name)
    #name = './data/sk2_dataset/categories_samples/'
    #get_audio_categories(name)
    #name = './data/youtube_test/categories_samples/'
    #get_audio_categories(name)
    #dataset_name = sys.argv[1]
    #get_audio_categories(dataset_name)
