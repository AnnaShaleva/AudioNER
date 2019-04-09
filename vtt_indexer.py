import webvtt
import time
import datetime
import os
from elasticsearch import Elasticsearch

import constants as const

def parse_subs(source_path):
    result = []
    for caption in webvtt.read(source_path):
        start = time.mktime(datetime.datetime.strptime(caption.start, "%H:%M:%S.%f").timetuple())
        end = time.mktime(datetime.datetime.strptime(caption.end, "%H:%M:%S.%f").timetuple())
        if (end - start > 0.02):
            line = {
                    'start': caption.start,
                    'end': caption.end,
                    'text': caption.text.split('\n')[1]
                    }
            result.append(line)
    return result

def index_subs_dataset(dataset_path):
    es = Elasticsearch()
    for subs_file in os.listdir(dataset_path):
        num = 1
        for line in parse_subs(os.path.join(dataset_path, subs_file)):
            es.index(index="subs_index", doc_type=subs_file, id=num, body=line)
            num += 1

if __name__=="__main__":
    dataset_folder = sys.argv[1]
    dataset_path = os.path.join(const.DATA_DIR, dataset_folder, "subs/")
#out_path = os.path.join(const.DATA_DIR, sys.argv[2])
    index_subs_dataset(dataset_path)
