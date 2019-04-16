import scipy.io.wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import sys

def detect_maxima(source):
    sr_value, x_value = scipy.io.wavfile.read(source)

    # Plot audio
    N = x_value.shape[0]
    L = N / sr_value
    plt.plot(np.arange(N) / sr_value, x_value)
    plt.title('Audio')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [?]')
    plt.savefig('audio.png')
    plt.clf()

    #Get spectrogram
    f, t, Sxx= signal.spectrogram(x_value,sr_value)

    #Plot spectrogram
    plt.pcolormesh(t, f, Sxx)
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram')
    plt.savefig('spectrogram.png')
    plt.clf()

    #Get maxima
    neighborhood_size = 20
    threshhold = 1000
    data_max = maximum_filter(Sxx, neighborhood_size)
    maxima = (Sxx == data_max)
    data_min = minimum_filter(Sxx, neighborhood_size)
    diff = ((data_max - data_min) > threshhold)
    maxima[diff == 0] = 0

    #Get maxima with erosion
    local_max = maximum_filter(Sxx, neighborhood_size) == Sxx
    background = (Sxx == 0)
    eroded_background = binary_erosion(background, generate_binary_structure(2, 2))
    eroded_maxima = local_max ^ eroded_background


    #Plot spectrogram
    im = plt.pcolormesh(t, f, maxima)
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram maxima')
    plt.savefig('spectrogram_maxima.png')
    plt.clf()

    # Plot eroded spectrogram
    im = plt.pcolormesh(t, f, eroded_maxima)
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram maxima')
    plt.savefig('eroded_spectrogram_maxima.png')
    plt.clf()

    return t, f, maxima

if __name__=='__main__':
    detect_maxima(sys.argv[1])