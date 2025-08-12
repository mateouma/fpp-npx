import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colormaps

def gen_filter(waveform, n, fs, truncate_idx=62, truncate_val=None, center=False):
    waveform = waveform[:truncate_idx]

    filter_t = np.zeros((n))
    filter_t[:truncate_idx] = waveform

    if truncate_val == None:
        truncate_val = waveform[-1]

    filter_t[truncate_idx:] = truncate_val

    if center:
        filter_t -= truncate_val

    wf_fft = np.fft.fft(filter_t) / n

    freq_idx = np.fft.fftfreq(n, 1/fs)
    sort_fftf = np.argsort(freq_idx)

    filter_f = wf_fft[sort_fftf]
    filter_psd = (np.abs(filter_f))**2

    freq_axis = np.linspace(-int(fs/2), int(fs/2), n)

    return filter_t, filter_f, filter_psd, freq_axis