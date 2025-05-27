import numpy as np
import matplotlib.pyplot as plt

from fppnpx.signalfuncs import load_signal, gen_all_channel_signals
from fppnpx.spectrafuncs import *

from plottools import ml_map

apath = "/Users/mateouma/Downloads/monkey datasets/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0_t0.exported.imec0.lf.bin"
cpath = "/Users/mateouma/Downloads/monkey datasets/cluster_info_task.tsv"
wpath = "/Users/mateouma/Downloads/monkey datasets/20230630_DLPFCwaveforms.mat"

fs = 2500
time_window = [100,150] # seconds within the recording

signal_dataset = load_signal(appath=apath, time_window=time_window, fs=fs, cipath=cpath, wfpath=wpath)
channel_signals = gen_all_channel_signals(signal_dataset, 300, False)

sec_time_binned_spectra = []

for tb in range(time_window[1] - time_window[0]):
    tb1 = tb * fs
    tb2 = (tb + 1) * fs
    print(tb1,tb2)

    channel_mt_spectra = []

    for chID in signal_dataset['channels']:
        chan_PSD,mt_freqs = multitaper_psd(channel_signals[f'ch{chID}'].time_series[tb1:tb2], fs=fs, thbp=2, start_time=time_window[0]+tb, verbose=False)
        channel_mt_spectra.append(chan_PSD)

    channel_mt_spectra = np.array(channel_mt_spectra)

    channel_mt_spectra_trunc = channel_mt_spectra[:,mt_freqs < 150]
    sec_time_binned_spectra.append(channel_mt_spectra_trunc)

tb_samp = np.random.choice(np.arange(time_window[1] - time_window[0]), 25)

sec_time_binned_spectra = np.array(sec_time_binned_spectra)[tb_samp]

channel_mt_spectra_mean = sec_time_binned_spectra.mean(axis=0)

ch_rel_power = channel_mt_spectra_mean / np.max(channel_mt_spectra_mean, axis=0)

unit_depths = np.array([signal_dataset['cluster_info'][signal_dataset['cluster_info']['cluster_id'] == u_id]['depth'] for u_id in signal_dataset['units']])
unit_depths = np.squeeze(unit_depths)

chan_idx = np.argsort(unit_depths)

plt.figure(figsize=(6,6))
plt.imshow(ch_rel_power[chan_idx], cmap=ml_map, aspect='auto')
plt.colorbar()
plt.show(block=True)