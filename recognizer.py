#Импорт библиотек и функций
import os
import sys

current_dir_path = os.path.dirname(os.path.realpath(__file__))
project_root_path = os.path.join(current_dir_path, os.pardir, os. pardir)
sys.path.insert(0, project_root_path)

from flask import Flask, request, jsonify
from werkzeug import secure_filename
from tensorflow.keras.models import load_model
import numpy as np

from audio_utils import convert_audio_to_wav
from audio_utils import apply_bandpass_filter
from audio_utils import normalize_volume
from audio_utils import audio_file_to_input_vector
from audio_utils import split_audio
from commands_utils import match_labels_with_command
from commands_utils import convert_commands_to_relative_coordinates
import constants as const

#Создание инстанса Flask
app = Flask(__name__)

#Загрузка обученной модели
model = load_model(const.MODEL_PATH)

#Задание маршрутов Flask
@app.route('/process_command', methods=['Post'])
def process_command():
    try:
        #Сохранение полученного аудио-файла
        audio = request.files['file']
        audio_path = os.path.join(const.AUDIO_SOURCE_PATH, secure_filename(audio.filename))
        audio.save(audio_path)

        #Предобработка аудио
        preprocessed_audio_path = preprocess_audio(audio_path)

        #Выделение команд из аудио-файла
        commands = get_commands(preprocessed_audio_path)

        #Преобразование команд в последовательность относительных координат точки
        relative_coordinates = convert_commands_to_relative_coordinates(commands)

        #Запись относительных координат в файл
        with open(const.COORDINATES_SOURCE_PATH, 'a+') as f:
            f.write('\n'.join('%s %s' % x for x in relative_coordinates))

        #Формирование ответа клиенту
        resp = jsonify({'commands': commands})
        resp.status_code = 200

        return resp

    except Exception as e:
        raise e

#Задание обработчика ошибок Flask
@app.errorhandler(Exception)
def exception_handler(error=None):
    message = {
                'status': 500,
                'message': 'Internal server error: ' + str(error),
            }
    resp = jsonify(message)
    resp.status_code = 500

    return resp

#Функция предобработки аудио
def preprocess_audio(audio_path):
    #Преобразование в .wav формат с заданными характеристиками
    wav_audio_path = os.path.join(const.DATA_PATH, 'audio.wav')
    convert_audio_to_wav(source=audio_path, sample_rate=16000, n_channels=1, byte_width=2, dest=wav_audio_path)

    #Нормализация громкости аудио
    normalized_audio_path = os.path.join(const.DATA_PATH, 'normalized_audio.wav')
    normalize_volume(source=wav_audio_path, level=-10, dest=normalized_audio_path)

    #Применение полосового частотного фильтра
    filtered_audio_path = os.path.join(const.DATA_PATH, 'filtered_audio.wav')
    apply_bandpass_filter(source=normalized_audio_path, lower_bound=250, upper_bound=3000, dest=filtered_audio_path)

    return filtered_audio_path

#Функция выделения команд из аудио
def get_commands(audio_path):
    #Разделение аудио-файла на семплы длиной 0.8 секунды без сдвига
    original_samples_paths = split_audio(source=audio_path, shift=.0, duration=const.SAMPLE_LENGTH)

    #Формирование входной последовательности матриц MFCC
    X_original = []
    for filename in original_samples_paths:
        X_original.append(audio_file_to_input_vector(source=filename, numcep=const.N_INPUT, numcontext=const.N_CONTEXT))

    #Предсказание классов и подклассов для входной последовательности
    Y_class_original, Y_subclass_original = model.predict(X_original)

    #Разделение аудио-файла на семплы длиной 0.8 секунды со сдвигом 0.4 секунды от начала аудио
    shifted_samples_paths = split_audio(source=audio_path, shift=0.4, duration=const.SAMPLE_LENGTH)

    #Формирование входной последовательности матриц MFCC
    X_shifted = []
    for filename in shifted_samples_paths:
        X_shifted.append(audio_file_to_input_vector(source=filename, numcep=const.N_INPUT, numcontext=const.N_CONTEXT))

    # Предсказание классов и подклассов для входной последовательности со сдвигом
    Y_class_shifted, Y_subclass_shifted = model.predict(X_shifted)

    #Формирование листа команд
    commands = []
    for i in range(len(X_original)):
        # Выбор классов и подклассов с лучшей точностью
        original_class_label = np.argmax(Y_class_original[i])
        original_subclass_label = np.argmax(Y_subclass_original[i])
        shifted_class_label = np.argmax(Y_class_shifted[i])
        shifted_subclass_label = np.argmax(Y_subclass_shifted[i])

        #Добавление команд в лист
        commands.append(match_labels_with_command(original_class_label, original_subclass_label))
        if (original_class_label != shifted_class_label) | (original_subclass_label != shifted_subclass_label):
            commands.append(match_labels_with_command(shifted_class_label, shifted_subclass_label))

    return commands


if __name__ == '__main__':
    #Запуск веб-приложения
    app.run(host='0.0.0.0', port = 5000, debug = True)