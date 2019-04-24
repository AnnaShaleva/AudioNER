from dataset_collector import get_times_of_spectr_maxima, cut_audio

from elasticsearch import Elasticsearch
import sys

def get_audio_categories(dataset_name):
    es = Elasticsearch(
            [const.ELASTIC_HOST]
            )
    with open(os.path.join(const.CATEGORIES_PATH, 'number.txt'), 'r') as f:
        count = 0
        for subcategory in f:
            matches = es.search(index=dataset_name,
                    body={
                        'query': {
                            'match': {
                                'text': str(count)
                                }
                            }
                        }
                    )
            for item in matches['hits']['hits']:
                cut_audio(item["_index"], item['_id'], item["_source"], 'number', subcategory.strip())
            count += 1

if __name__=="__main__":
    dataset_name = sys.argv[1]
    get_audio_categories(dataset_name)
