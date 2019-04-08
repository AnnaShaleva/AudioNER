import pafy
import os
import sys

import constants as const

def download_and_preprocess_data(urls_source_file, out_path):
    if not os.path.exists(urls_source_file):
        raise ValueError("URLs source file does not exist.")

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    with open (urls_source_file, "r") as f:
        for url in f:
            p = pafy.new(url)
            ba = p.getbestaudio()
            filename = ba.download(out_path)
            print(filename + " was downloaded")

if __name__=="__main__":
    source_file = sys.argv[1]
    out_path = os.path.join(const.DATA_DIR, sys.argv[2])
    download_and_preprocess_data(source_file, out_path)
