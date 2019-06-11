# AudioNER

This repo contains the large corpora of russian-language speech commands data gathered from different sources. It also includes scripts for data handling, pre-processing and features extraction. Moreover, you can find here several Deep CNN models for the Keywords-Spotting task.

One of the trained models was integrated to the human-machine interaction system with voice interface, which is able to record audio command, send it to server, recognize keywords from classes `Number` and `Direction` and plot the result of the command. Here you can fined parts of this system.
## Repo structure

The repository is organised in the following way:

| Folder | Description | Tools |
| ----- | ----------- | -----|
| [data](https://github.com/AnnaShaleva/AudioNER/tree/master/data) | Contains several datasets for KWS task gathered from different sources, e.g. YouTube channels ([Heads and Tails](https://github.com/AnnaShaleva/AudioNER/blob/master/data/ht_old_urls.txt), [Echo of Moskow](https://github.com/AnnaShaleva/AudioNER/blob/master/data/echo_of_moscow_urls) and others) and [Azure TTS Service](https://azure.microsoft.com/en-us/services/cognitive-services/text-to-speech/). [import_youtube.py](https://github.com/AnnaShaleva/AudioNER/blob/master/import_youtube.py) contains methods fordownloading and preprocessing data.| `pafy`, `ffmpeg`, `youtube-dl` |
| [categories](https://github.com/AnnaShaleva/AudioNER/tree/master/categories) | Contains `.txt` files with keywords groupped into classes: `Name`, `Direction`, `Number`, `Confirm`, `Action` | Classes were chosen with the help of [this](https://arxiv.org/abs/1804.03209) paper. |
| [NNI_models](https://github.com/AnnaShaleva/AudioNER/tree/master/NNI_models) | Contains folders for each NNI hyperparams optimisation experiment with model description in `model.py`, configuration description in `config.yml` and search space in `searchspace.json`. | `Microsoft NNI`, `TensorFlow`, `Keras`|
| [vkr_models](https://github.com/AnnaShaleva/AudioNER/tree/master/vkr_models) | Contains 6 experiments for training models CNN, DS-CNN and M-CNN on datasets `yt_tts_clean` and `yt_tts_augmented`. Also includes saved models. | `Python 3.6`, `Tensorflow`, `Keras` |
| [recognizer.py](https://github.com/AnnaShaleva/AudioNER/blob/master/recognizer.py) | Module-recognizer for extraction commands from audio. Uses model `M-CNN` with best accuracy to recognize keywords from audio semples. | `Python 3.6`, `Flask`, `M-CNN` |
| [interpreter.py](https://github.com/AnnaShaleva/AudioNER/blob/master/interpreter.py) | Module-interpreter reads relative coordinates from file and plot the trajectory of a point. | `Python 3.6`, `matplotlib` |

Also at the root directory of the repo you can fined utils for data pre-processing and handling.
