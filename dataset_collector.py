from elasticsearch import Elasticsearch
import sys
import os
import time
import datetime
import subprocess

import constants as const

def cut_audio(dataset_name, source_id, source, category, subcategory):
    audiofile = os.path.splitext(os.path.splitext(source['filename'])[0])[0] + ".m4a"
    audio_path = os.path.join(const.DATA_PATH, dataset_name + "/", "audio/", audiofile)

    if not os.path.isfile(audio_path):
        return

    subcategory_samples_path = os.path.join(const.DATA_PATH, dataset_name + "/", "categories_samples/", category + "/", subcategory + "/")
    if not os.path.isdir(subcategory_samples_path):
        os.makedirs(subcategory_samples_path)
    
    sample_path = os.path.join(subcategory_samples_path, str(source_id) + ".wav")

    p = subprocess.Popen(["ffmpeg",
        "-i", audio_path,
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        "-ss", source['start'],
        "-to", source['end'],
        sample_path        
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    out, err = p.communicate()

    if p.returncode != 0:
        raise Exception("Failed to cut audio: %s" % str(err))
    
    print("%s was saved" % sample_path)

def get_audio_categories(dataset_name):
    es = Elasticsearch(
            [const.ELASTIC_HOST]
            )
    for category_file in os.listdir(const.CATEGORIES_PATH):
        category = os.path.splitext(category_file)[0]
        with open(os.path.join(const.CATEGORIES_PATH, category_file), 'r') as f:
            for subcategory in f:
                matches = es.search(index=dataset_name,
                        body={ 
                            'query': {
                                'fuzzy': {
                                    'text': {
                                        'value': subcategory.strip(),
                                        'boost': 1.0,
                                        'fuzziness': 1,
                                        'prefix_length': 0,
                                        'max_expansions': 100
                                        }
                                    }
                                }
                            }
                        )
                for item in matches['hits']['hits']:
                    cut_audio(item["_index"], item['_id'], item["_source"], category, subcategory.strip())

if __name__=="__main__":
    dataset_name = sys.argv[1]
    get_audio_categories(dataset_name)
