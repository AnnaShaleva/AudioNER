import scipy.io.wavfile
from scipy.io.wavfile import read
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion


sr_value, x_value = scipy.io.wavfile.read("cut.wav")

f, t, Sxx= signal.spectrogram(x_value,sr_value)

neighborhood_size = 5
threshhold = 0.1
data_max = maximum_filter(Sxx, neighborhood_size)
maxima = (Sxx == data_max)
data_min = minimum_filter(Sxx, neighborhood_size)
diff = ((data_max - data_min) > threshhold)
maxima[diff == 0] = 0

plt.pcolormesh(t, f, maxima)
plt.savefig('maxima.png')

plt.pcolormesh(t, f, Sxx)
neighbors = generate_binary_structure(4, 4)
local_max = maximum_filter(Sxx, footprint=neighbors) == Sxx
print(local_max)
background = (Sxx < 0.1)
print(background)
eroded_background = binary_erosion(background, structure=neighbors, border_value=1)
print(eroded_background)
detected_peaks = local_max ^ eroded_background

plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.savefig('Testplot.png')
plt.pcolormesh(t, f, detected_peaks)
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.savefig('Testplot_peaks.png')

N = x_value.shape[0]
L = N / sr_value

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.arange(N) / sr_value, x_value)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude [?]')

fig.savefig('audio.png')

