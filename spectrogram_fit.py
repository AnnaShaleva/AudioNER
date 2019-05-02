import scipy.io.wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filtermmm
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

path = 'cut.wav'
sr_value, x_value = scipy.io.wavfile.read(out_path)
f, t, Sxx = signal.spectrogram(x_value, sr_value)