import pafy
import os
import sys

import constants as const

def download_and_preprocess_data(urls_source_file, out_path):
    if not os.path.exists(urls_source_file):
        raise ValueError("URLs source file does not exist.")

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    audio_out_path = os.path.join(out_path, "audio/")
    if not os.path.isdir(audio_out_path):
        os.makedirs(audio_out_path)

    with open (urls_source_file, "r") as f:
        for url in f:
            try:
                playlist = pafy.get_playlist(url)
                for item in playlist['items']:
                    audio = item['pafy'].getbestaudio()
                    filename = os.path.join(audio_out_path, item['pafy'].videoid + "." + audio.extension)
                    result = audio.download(filename)
                    print(result + " was downloaded")
            except:
                continue

if __name__=="__main__":
    source_file = sys.argv[1]
    out_path = os.path.join(const.DATA_DIR, sys.argv[2])
    download_and_preprocess_data(source_file, out_path)
