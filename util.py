import sys
import os
import constants as const


def match_subs_audio(dataset_name):
    print('######## MATCH SUBS AUDIO #############')
    subs_path = (os.path.join(const.DATA_PATH, dataset_name + '/', 'subs/'))
    audio_path = (os.path.join(const.DATA_PATH, dataset_name + '/', 'audio/'))
    count = 0
    for sub_file in os.listdir(subs_path):
        video_id = os.path.splitext(os.path.splitext(sub_file)[0])[0]
        #print(video_id)
        audio_file = os.path.join(audio_path, video_id + '.m4a')
        if not os.path.isfile(audio_file):
            print('Remove:' + sub_file)
            count +=1
            #os.remove(os.path.join(subs_path, sub_file))

def match_audio_subs(dataset_name):
    print('######## MATCH AUDIO SUBS #############')
    subs_path = (os.path.join(const.DATA_PATH, dataset_name + '/', 'subs/'))
    audio_path = (os.path.join(const.DATA_PATH, dataset_name + '/', 'audio/'))

    for audio_file in os.listdir(audio_path):
        video_id = os.path.splitext(audio_file)[0]
        #print(video_id)
        count = 0
        sub_file = os.path.join(subs_path, video_id + '.ru.vtt')
        if not os.path.isfile(sub_file):
            print('Remove:' + audio_file)
            count+=1
            #os.remove(os.path.join(audio_path, audio_file))

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    match_subs_audio(dataset_name)
    match_audio_subs(dataset_name)
