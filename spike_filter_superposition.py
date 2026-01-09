import numpy as np
import pandas as pd
import mat73 as mt
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from scipy import stats

from fppnpx.preprocessing import read_session, load_waveforms, read_bin
from fppnpx.ChannelSignal import ChannelSignal
from fppnpx.plottools import MATMAP, WAVEMAP_PAL2, MISC6_PAL
from fppnpx.filters import gen_filter
from fppnpx.spectral import multitaper_spectrum, multitaper_spectrogram, spectrum_trunc, spectrum_interp

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

imecpath = "/Users/mateouma/Desktop/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0"

time_window = [204,209]
npxsession = read_session(time_window=time_window, imec_path=imecpath)

waveformpath = "/Users/mateouma/Downloads/monkey datasets/20230630_DLPFCwaveforms.mat"
waveform_dataset, waveform_means = load_waveforms(waveformpath, npxsession)

tw = 3.5
n_taps = 2*tw - 1
fs = 30000

ch163 = ChannelSignal(163, session_dataset=npxsession, bandtype='AP', waveform_dataset=waveform_dataset, notch_filt=1, bandpass_filt=None, time_halfbandwidth_product=tw, verbose=False)
ch208 = ChannelSignal(208, session_dataset=npxsession, bandtype='AP', waveform_dataset=waveform_dataset, notch_filt=1, bandpass_filt=None, time_halfbandwidth_product=tw, verbose=False)

def plot_filter_superposition(channel_signal, unit_list, mode='isolated', truncate_idx=62):
    # indices for convolve windows
    filter_add_length = truncate_idx - 20
    end_idx = -(filter_add_length - 1)

    # make isolated spike time series
    spikes_conv_list = []
    spikes_mask_list = []

    for i,unit in enumerate(unit_list):
        unit_spike_train = np.zeros_like(channel_signal.time_series)
        unit_spike_train[npxsession['cluster_spike_times'][unit]] = 1
        unit_spikes_conv = np.convolve(unit_spike_train, waveform_dataset[unit]['time_filter'][:truncate_idx])[20:end_idx]
        spikes_mask_list.append(unit_spikes_conv == 0)
        spikes_conv_list.append(unit_spikes_conv)

    if mode == 'isolated':
        total_time_series = np.sum(np.array(spikes_conv_list), axis=0)
    elif mode == 'raw':
        total_time_series = channel_signal.time_series

    # calculate spectrum of time series
    mt_PSD,mt_freqs = multitaper_spectrum(total_time_series, fs=fs, time_halfbandwidth_product=tw, start_time=time_window[0])

    # theoretical spectra
    theoretical_freq_axis = waveform_dataset[unit_list[0]]['filter_freq_axis']
    theoretical_spectrum_list = []
    for unit in unit_list:
        unit_theoretical_spectrum = waveform_dataset[unit]['filter_spectrum'] * waveform_dataset[unit]['firing_rate']
        theoretical_spectrum_list.append(unit_theoretical_spectrum)
    theoretical_spectrum = np.sum(np.array(theoretical_spectrum_list), axis=0)

    # QQ plots
    freq_range = (300,5000)

    mt_PSD_trunc,mt_freqs_trunc = spectrum_trunc(mt_freqs,mt_PSD,freq_range)
    theor_spk_PSD_trunc,theor_freqs_trunc = spectrum_trunc(theoretical_freq_axis, theoretical_spectrum,freq_range)

    theor_spk_PSD_interp = spectrum_interp(mt_freqs_trunc, theor_freqs_trunc, theor_spk_PSD_trunc)

    Z_f = np.sort((n_taps * mt_PSD_trunc) / theor_spk_PSD_interp)
    n = Z_f.size

    theor_quantiles = stats.gamma.ppf((np.arange(1, n + 1) - 0.5) / n, a=n_taps)

    # QQ (300 - 1000 Hz)
    freq_range2 = (300,1000)

    mt_PSD_trunc2,mt_freqs_trunc2 = spectrum_trunc(mt_freqs,mt_PSD,freq_range2)
    theor_spk_PSD_trunc2,theor_freqs_trunc2 = spectrum_trunc(theoretical_freq_axis,theoretical_spectrum,freq_range2)
    theor_spk_PSD_interp2 = spectrum_interp(mt_freqs_trunc2, theor_freqs_trunc2, theor_spk_PSD_trunc2)

    Z_f2 = np.sort((n_taps * mt_PSD_trunc2) / theor_spk_PSD_interp2)
    n2 = Z_f2.size

    theor_quantiles_2 = stats.gamma.ppf((np.arange(1, n2 + 1) - 0.5) / n2, a=n_taps)

    gridspec = {'width_ratios': [1, 1, 0.4, 0.4]}
    fig, ax = plt.subplots(1,4, figsize=(10,2), gridspec_kw=gridspec)

    # time series
    ax[0].plot(channel_signal.time_axis, total_time_series, color='k', linewidth=0.8)
    for i,unit in enumerate(unit_list):
        time_series_excerpt = total_time_series.copy()
        time_series_excerpt[spikes_mask_list[i]] = np.nan
        ax[0].plot(channel_signal.time_axis, time_series_excerpt, color=MISC6_PAL[i], linewidth=0.8)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Voltage (mV)")
    ax[0].set_xlim((203.5, 209.5))
    ax[0].set_xticks([204,206,208,209])
    ax[0].set_xticklabels([204,206,208,''])
    ax[0].set_yticks([-125, 0, 125])
    ax[0].set_ylim((-150, 150))
    ax[0].spines['bottom'].set_bounds(204, 209)
    ax[0].spines['left'].set_bounds(-125, 125)

    # spectrum
    ax[1].semilogy(mt_freqs, mt_PSD, color='k', linewidth=0.6)
    for i,unit in enumerate(unit_list):
        ax[1].semilogy(theoretical_freq_axis, theoretical_spectrum_list[i], color=MISC6_PAL[i], linewidth=2)
    ax[1].set_xlim(250, 5050)
    ax[1].set_ylim(6e-9, 1)
    ax[1].set_xticks([300, 1000, 2500, 5000])
    ax[1].set_yticks([1e-8,1e-6,1e-4,1e-2,1])
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Log Power")
    ax[1].spines['left'].set_bounds(1e-8, 1)
    ax[1].spines['bottom'].set_bounds(300, 5e3)

    # QQ
    ax[2].scatter(theor_quantiles, Z_f, marker='x', c='k', s=1)
    ax[2].plot(np.arange(-100,100), np.arange(-100,100), color='red', linewidth=0.5)
    ax[2].set_xlim((-1, 21))
    ax[2].set_ylim((-1, 21))
    ax[2].set_xlabel("Theoretical quantiles")
    ax[2].set_ylabel("Sample quantiles")
    ax[2].set_xticks([0,10,20])
    ax[2].spines['bottom'].set_bounds(0, 20)
    ax[2].spines['left'].set_bounds(0, 20)

    # QQ (300 - 1000 Hz)
    ax[3].scatter(theor_quantiles_2, Z_f2, marker='x', c='k', s=1)
    ax[3].plot(np.arange(-100,100), np.arange(-100,100), color='red', linewidth=0.5)
    ax[3].set_xlim((-1, 21))
    ax[3].set_ylim((-1, 21))
    ax[3].set_xlabel("Theoretical quantiles\n (300-1000 Hz)")
    ax[3].set_ylabel("Sample quantiles\n (300-1000 Hz)")
    ax[3].set_xticks([0,10,20])
    ax[3].spines['bottom'].set_bounds(0, 20)
    ax[3].spines['left'].set_bounds(0, 20)

    fig.tight_layout()

    return fig

ch163_isolated = plot_filter_superposition(ch163, [296], mode='isolated')
ch163_isolated.savefig("figs/ch163_isolated_superposition.svg")

ch163_raw = plot_filter_superposition(ch163, [296], mode='raw')
ch163_raw.savefig("figs/ch163_raw_superposition.svg")

ch208_isolated = plot_filter_superposition(ch208, [378, 379, 374, 375, 380, 383], mode='isolated')
ch208_isolated.savefig("figs/ch208_isolated.svg")

ch208_raw = plot_filter_superposition(ch208, [378, 379, 374, 375, 380, 383], mode='raw')
ch208_raw.savefig("figs/ch208_raw.svg")