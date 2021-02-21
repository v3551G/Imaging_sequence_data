# -*- coding: utf-8 -*-
"""
@author: masterqkk, masterqkk@outlook.com
Environment:
    python: 3.6
    Pandas: 1.0.3
    matplotlib: 3.2.1
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scisig
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyts.datasets import load_gunpoint

if __name__ == '__main__':
    X, _, _, _ = load_gunpoint(return_X_y=True)

    fs = 10e3  # sampling frequency
    N = 1e5  # 10 s 1signal
    amp = 2 * np.sqrt(2)
    time = np.arange(N) / float(fs)
    mod = 500 * np.cos(2 * np.pi * 0.25 * time)
    carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
    noise_power = 0.01 * fs / 2
    noise = np.random.normal(loc=0.0, scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time / 5)
    x = carrier + noise  # signal with noise

    per_seg_length = 1000 # window length
    f, t, Zxx = scisig.stft(x, fs, nperseg=per_seg_length, noverlap=0, nfft=per_seg_length, padded=False)
    print('Zxx.shaope: {}'.format(Zxx.shape))

    plt.figure()
    plt.suptitle('STFT')
    ax1 = plt.subplot(211)
    ax1.plot(x)
    plt.title('signal with noise')

    ax2 = plt.subplot(212)
    x = np.abs(Zxx)
    ax2.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
    plt.title('STFT Magnitude')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time [sec]')
    plt.show()

