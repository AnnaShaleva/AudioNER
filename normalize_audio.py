# -*- coding: utf-8 -*-

import subprocess
import wave

import numpy as np

def loud_norm(in_path, out_path):
    # ffmpeg-normalize audio.wav -o out.wav
    # ffmpeg -i audio.wav -filter:a loudnorm loudnorm.wav
    p = subprocess.Popen(["ffmpeg", "-y",
        "-i", in_path,
        "-af", "loudnorm",
        out_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out, err = p.communicate()

    if p.returncode != 0:
        raise Exception("Failed to loudnorm: %s" % str(err))


def apply_bandpass_filter(in_path, out_path, lowpass=8000, highpass=50):
    # ffmpeg -i input.wav -acodec pcm_s16le -ac 1 -ar 16000 -af lowpass=3000,highpass=200 output.wav
    p = subprocess.Popen(["ffmpeg", "-y",
        "-acodec", "pcm_s16le",
         "-i", in_path,
         "-acodec", "pcm_s16le",
         "-ac", "1",
         "-af", "lowpass=%i,highpass=%i" % (lowpass, highpass),
         "-ar", "16000",
         out_path
         ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out, err = p.communicate()

    if p.returncode != 0:
        raise Exception("Failed to apply bandpass filter: %s" % str(err))

def correct_volume(in_path, out_path, db=-2):
    # ffmpeg -i audio.wav -filter:a "volume=-2dB" loudnorm_vol_set.wav
    p = subprocess.Popen(["ffmpeg", "-y",
         "-i", in_path,
         "-filter:a", "volume=%idB" % (db),
         out_path
         ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out, err = p.communicate()

    if p.returncode != 0:
        raise Exception("Failed to correct volume: %s" % str(err))