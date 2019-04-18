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

def get_times_of_spectr_maxima(source_path):
    sr_value, x_value = scipy.io.wavfile.read(source_path)
    f, t, Sxx = signal.spectrogram(x_value, sr_value)

    data_max = maximum_filter(Sxx, const.NEIGHBORHOOD_SIZE)
    maxima = (Sxx == data_max)
    data_min = minimum_filter(Sxx, const.NEIGHBORHOOD_SIZE)
    diff = ((data_max - data_min) > const.AMPLITUDE_DIFFERENCE_TRASHHOLD)
    maxima[diff == 0] = 0
    maxima_indexes = np.where(maxima > 0)
    result = []
    for i in range(len(maxima_indexes[0])):
        if f[maxima_indexes[0][i]] < const.FREQUENCY_TRESHHOLD:
            result.append(t[maxima_indexes[1][i]])

    return result

def cut_audio(dataset_name, source_id, source, category, subcategory):
    audiofile = os.path.splitext(os.path.splitext(source['filename'])[0])[0] + ".m4a"
    audio_path = os.path.join(const.DATA_PATH, dataset_name + "/", "audio/", audiofile)

    if not os.path.isfile(audio_path):
        return

    subcategory_samples_path = os.path.join(const.DATA_PATH, dataset_name + "/", "categories_samples/", category + "/", subcategory + "/")
    if not os.path.isdir(subcategory_samples_path):
        os.makedirs(subcategory_samples_path)

    sample_path = os.path.join(subcategory_samples_path, str(source_id) + ".wav")
    if os.path.isfile(sample_path):
        return

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
        raise Exception("Failed to cut audio at step 1: %s" % str(err))

    maxima_time = get_times_of_spectr_maxima(sample_path)
    start_time = datetime.datetime.strptime(source['start'], '%H:%M:%S.%f')
    start = datetime.timedelta(hours=start_time.hour, minutes=start_time.minute, seconds=start_time.second, microseconds=start_time.microsecond) 
    if (len(maxima_time) > 0) & (start > datetime.timedelta(seconds=0.1)):
        start += datetime.timedelta(seconds=min(maxima_time)) - datetime.timedelta(seconds=0.1)
        print("Start was changed")
    
    print(source['start'])
    print(start)

    p = subprocess.Popen(["ffmpeg",
                          "-y",
                          "-i", audio_path,
                          "-acodec", "pcm_s16le",
                          "-ac", "1",
                          "-ar", "16000",
                          "-ss", str(start),
                          "-t", "0.8",
                          sample_path
                          ],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    out, err = p.communicate()

    if p.returncode != 0:
        raise Exception("Failed to cut audio at step 2: %s" % str(err))

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
