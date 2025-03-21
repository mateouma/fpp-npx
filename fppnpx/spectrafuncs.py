import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

from spectral_connectivity import Multitaper, Connectivity

def multitaper_psd(time_series, fs, n_tapers, start_time=0.0):
    multitaper = Multitaper(
            time_series,
            sampling_frequency=fs,
            n_tapers=n_tapers,
            start_time=start_time,
        )
    connectivity = Connectivity.from_multitaper(multitaper)
    multitaper_psd = connectivity.power().squeeze()
    multitaper_frequencies = connectivity.frequencies

    return multitaper_psd, multitaper_frequencies

def multitaper_spectrogram(time_series, fs, time_halfbandwidth_product, window_duration, window_step, start_time=0.0):
    multitaper = Multitaper(
        time_series,
        sampling_frequency=fs,
        time_halfbandwidth_product=time_halfbandwidth_product,
        time_window_duration=window_duration,
        time_window_step=window_step,
        start_time=start_time,
    )
    connectivity = Connectivity.from_multitaper(multitaper)
    
    return connectivity

def spectrum_trunc(freqs, spectrum, freq_range):
    """
    'Filters' a power spectra
    """
    freq_min,freq_max = freq_range
    trunc_idx = np.where(
        (freqs >= freq_min) & (freqs <= freq_max)
    )
    freqs_trunc = freqs[trunc_idx]
    spectrum_trunc = spectrum[trunc_idx]

    return spectrum_trunc, freqs_trunc


def spectrum_interp(reference_freqs, data_freqs, spectrum):
    """
    Interpolates power spectra
    """
    spectrum_interpolated = np.interp(
        reference_freqs,
        data_freqs,
        spectrum
    )

    return spectrum_interpolated

def spectrum_smooth(spec, freqs, win_len_1=100, win_len_2=1000, split_idx=100, order=4):
    spec_split_1 = spec[freqs < split_idx]
    spec_split_2 = spec[freqs >= split_idx]

    spec_filt_1 = sig.savgol_filter(spec_split_1, win_len_1, order)
    spec_filt_2 = sig.savgol_filter(spec_split_2, win_len_2, order)

    return np.concatenate((spec_filt_1, spec_filt_2))

def plot_spectrum(x, fs, time_halfbandwidth_product=4, n_tapers=7, start_time=0.0, axes='loglog'):
    mt_psd, mt_freqs = multitaper_psd(x, fs, n_tapers, start_time)

    plt.figure(dpi=300)
    if axes=='loglog':
        plt.loglog(mt_freqs, mt_psd, linewidth=0.6, color='k')
    else:
        plt.plot(mt_freqs, mt_psd, linewidth=0.6, color='k')
    #plt.xlim((0,2000))

def plot_spectrogram(x, fs, spk_train=None, time_halfbandwidth_product=4, window_duration=0.05, window_step=0.025, start_time=0.0):
    fig, ax = plt.subplots()
    connectivity = multitaper_spectrogram(x, fs, time_halfbandwidth_product, window_duration, window_step, start_time=0.0)
    im = ax.pcolormesh(
        connectivity.time,
        connectivity.frequencies,
        connectivity.power().squeeze().T,
        cmap="viridis",
        shading="auto",
    )
    
    plt.ylim((0, 2000))

    if spk_train is not None:
        plt.scatter(spk_train / fs, np.repeat(250, len(spk_train)), color='red', marker='^', s=3)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Time")