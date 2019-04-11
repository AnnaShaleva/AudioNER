import webvtt
import time
import datetime
import sys
import re
import os
from elasticsearch import Elasticsearch

import constants as const

def parse_subs_into_line_tockens_list(source_path):
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

def parse_subs_into_word_tockens_list(source_path, subs_file):
    result = []
    for caption in webvtt.read(source_path):
        arr = caption.raw_text.split('\n')
        if(len(arr) > 1):
            line = arr[1]
            line = re.sub('[<][\/]?[c][^<]*[>]', "", line)
            line = line.replace(" ", "").lower()
            if line:
                line = "<" + caption.start + ">" + line + "<" + caption.end + ">"
                tockens = list(filter(None, re.split('[<>]', line)))
                count = 0
                while count < len(tockens) - 2:
                    start = tockens[count]
                    text = tockens[count + 1]
                    end = tockens[count + 2]
                    toAppend = {
                            'filename': subs_file,
                            'start': start,
                            'end': end,
                            'text': text
                            }
                    result.append(toAppend)
                    count += 2
    return result

def index_subs_dataset(dataset_name):
    dataset_path = os.path.join(const.DATA_DIR, dataset_name + "/", "subs/")
    es = Elasticsearch(
            [const.ELASTIC_HOST]
            )
    num = 1
    for subs_file in os.listdir(dataset_path):
        for element in parse_subs_into_word_tockens_list(os.path.join(dataset_path, subs_file), subs_file):
            es.index(index=dataset_name, doc_type="subs", id=num, body=element)
            num += 1
            print(element)

if __name__=="__main__":
    dataset_name = sys.argv[1]
    #out_path = os.path.join(const.DATA_DIR, sys.argv[2])
    index_subs_dataset(dataset_name)
