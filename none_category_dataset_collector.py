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
import contextlib
import wave

import constants as const


def cut_audio(dataset_name):
    print("################## STARTED CUTTING #################")
    source_path = os.path.join(const.DATA_PATH, dataset_name + "/", "categories_samples/", "none/", "tmp/")
    if not os.path.isdir(source_path):
        return

    category_samples_path = os.path.join(const.DATA_PATH, dataset_name + '/', "categories_samples/", "none/")
    if not os.path.isdir(category_samples_path):
        os.makedirs(category_samples_path)

    for audio_file in os.listdir(source_path):
        audio_path = os.path.join(source_path, audio_file)
        start = datetime.timedelta(seconds=1)
        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        duration = datetime.timedelta(seconds=duration)
        name_count = len([name for name in os.listdir(category_samples_path) if
                          os.path.isfile(os.path.join(category_samples_path, name))])
        while start < (duration - datetime.timedelta(seconds=4)):
            p = subprocess.Popen(["ffmpeg",
                                  "-i", audio_path,
                                  "-ss", str(start),
                                  "-t", "0.8",
                                  category_samples_path + str(name_count) + ".wav"
                                  ],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)

            out, err = p.communicate()

            if p.returncode != 0:
                raise Exception("END OF AUDIO : %s" % str(err))

            print(str(name_count) + " was saved")
            name_count += 1
            start += datetime.timedelta(seconds=3)
           


def clean_audio(dataset_name, source_file, periods_to_rm):
    source_audio_path = os.path.join(const.DATA_PATH, dataset_name + "/", "audio/", source_file)
    if not os.path.isfile(source_audio_path):
        return

    category_samples_path = os.path.join(const.DATA_PATH, dataset_name + "/", "categories_samples/", "none/" + "tmp/")
    if not os.path.isdir(category_samples_path):
        os.makedirs(category_samples_path)

    name_count = len([name for name in os.listdir(category_samples_path) if os.path.isfile(os.path.join(category_samples_path, name))])
    
    if not periods_to_rm:
        p = subprocess.Popen(["ffmpeg",
                             "-i", source_audio_path,
                             "-acodec", "pcm_s16le",
                             "-ac", "1",
                             "-ar", "16000",
                             category_samples_path + str(name_count) + ".wav"
                             ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

        out, err = p.communicate()

        if p.returncode != 0:
            raise Exception("Failed to save whole audio : %s" % str(err))
        print(source_audio_path + ": Nothing found")
        return

    periods_to_rm.sort()

    print("Source: " + source_audio_path)
    print("periods_to_rm.length: " + str(len(periods_to_rm)))
    outpath = category_samples_path + str(name_count) + '.wav'
    p = subprocess.Popen(["ffmpeg",
                          "-i", source_audio_path,
                          "-acodec", "pcm_s16le",
                          "-ac", "1",
                          "-ar", "16000",
                          "-ss", "0",
                          "-to", periods_to_rm[0][0],
                          outpath
                          ],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    out, err = p.communicate()

    if p.returncode != 0:
        #raise Exception("Failed to cut audio at zero step:" % str(err))
        pass
    print("Rm first period")

    if len(periods_to_rm) > 1:
        for i in range(1, len(periods_to_rm)):
            name_count = len([name for name in os.listdir(category_samples_path) if os.path.isfile(os.path.join(category_samples_path, name))])
 
            p = subprocess.Popen(["ffmpeg",
                                  "-i", source_audio_path,
                                  "-acodec", "pcm_s16le",
                                  "-ac", "1",
                                  "-ar", "16000",
                                  "-ss", periods_to_rm[i - 1][1],
                                  "-to", periods_to_rm[i][0],
                                  category_samples_path + str(name_count) + ".wav"
                                  ],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)

            out, err = p.communicate()

            if p.returncode != 0:
                #raise Exception("Failed to cut audio at last step: %s" % str(err))
                pass
            print("Rm period " + str(i))

    name_count = len([name for name in os.listdir(category_samples_path) if os.path.isfile(os.path.join(category_samples_path, name))])
 
    p = subprocess.Popen(["ffmpeg",
                          "-i", source_audio_path,
                          "-acodec", "pcm_s16le",
                          "-ac", "1",
                          "-ar", "16000",
                          "-ss", periods_to_rm[len(periods_to_rm) - 1][1],
                          category_samples_path + str(name_count) + ".wav"
                          ],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    out, err = p.communicate()

    if p.returncode != 0:
        #raise Exception("Failed to cut audio at last step: %s" % str(err))
        pass

    print("Rm last period")

def get_none_audio_category(dataset_name):
    es = Elasticsearch(
        [const.ELASTIC_HOST]
    )
    for audio_file in os.listdir(os.path.join(const.DATA_PATH, dataset_name + "/", "audio/")):
        periods_to_rm = []
        video_id = os.path.splitext(audio_file)[0]
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
                        if item["_source"]["filename"] == video_id + ".ru.vtt":
                            periods_to_rm.append((item["_source"]["start"], item["_source"]["end"], subcategory))
        clean_audio(dataset_name, audio_file, periods_to_rm) 
        print('################################# CLEANED #############################') 
    cut_audio(dataset_name)

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    get_none_audio_category(dataset_name)
    # dataset_name = "none_dataset"
    # cut_audio(dataset_name)
