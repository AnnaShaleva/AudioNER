from elasticsearch import Elasticsearch
import sys
import os
import time
import datetime
import subprocess
import scipy
from scipy.ndimage.filters import maximum_filter, minimum_filter
import numpy as np
import scipy.io.wavfile
from scipy import signal

import constants as const

def cut_audio(dataset_name, source_file, periods_to_rm):
    source_audio_path = os.path.join(const.DATA_PATH, dataset_name + "/", "audio/", source_file)

    if not os.path.isfile(source_audio_path):
        return

    category_samples_path = os.path.join(const.DATA_PATH, dataset_name + "/", "categories_samples/", "none/" + "tmp/")
    if not os.path.isdir(category_samples_path):
        os.makedirs(category_samples_path)

    name_count = len([name for name in os.listdir(category_samples_path) if os.path.isfile(name)])

    if not periods_to_rm:
        p = subprocess.Popen(["ffmpeg",
                              "-i", source_audio_path,
                              "-acodec", "pcm_s16le",
                              "-ac", "1",
                              "-ar", "16000",
                              category_samples_path + name_count + ".wav"
                              ],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

        out, err = p.communicate()

        if p.returncode != 0:
            raise Exception("Failed to cut audio at step 1: %s" % str(err))

        return

    periods_to_rm.sort()

    print(source_audio_path)
    print(periods_to_rm)
    # p = subprocess.Popen(["ffmpeg",
    #                       "-i", source_audio_path,
    #                       "-acodec", "pcm_s16le",
    #                       "-ac", "1",
    #                       "-ar", "16000",
    #                       "-ss", "00.00.00",
    #                       "-to", periods_to_rm[0]['start'],
    #                       category_samples_path + name_count + ".wav"
    #                       ],
    #                      stdout=subprocess.PIPE,
    #                      stderr=subprocess.PIPE)
    #
    # out, err = p.communicate()
    #
    # if p.returncode != 0:
    #     raise Exception("Failed to cut audio at step 1: %s" % str(err))
    #
    # name_count += 1
    #
    # p = subprocess.Popen(["ffmpeg",
    #                       "-i", source_audio_path,
    #                       "-acodec", "pcm_s16le",
    #                       "-ac", "1",
    #                       "-ar", "16000",
    #                       "-ss", periods_to_rm[len[periods_to_rm] - 1]['end'],
    #                       category_samples_path + name_count + ".wav"
    #                       ],
    #                      stdout=subprocess.PIPE,
    #                      stderr=subprocess.PIPE)
    #
    # out, err = p.communicate()
    #
    # if p.returncode != 0:
    #     raise Exception("Failed to cut audio at step 1: %s" % str(err))



def get_none_audio_category(dataset_name):
    es = Elasticsearch(
        [const.ELASTIC_HOST]
    )
    for audio_file in os.listdir(os.path.join(const.DATA_PATH, dataset_name + "/", "audio/")):
        periods_to_rm = []
        id = os.path.splitext(audio_file)
        for category_file in os.listdir(const.CATEGORIES_PATH):
            category = os.path.splitext(category_file)[0]
            with open(os.path.join(const.CATEGORIES_PATH, category_file), 'r') as f:
                for subcategory in f:
                    matches = es.search(index=dataset_name,
                                        body={
                                            'query': {
                                                'match': {
                                                    'filename' : id + ".ru.vtt"
                                                },
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
                        periods_to_rm.append({'start': item["_source"]["start"], 'end': item["_source"]["end"]})
        cut_audio(dataset_name, audio_file, periods_to_rm)

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    get_none_audio_category(dataset_name)
