import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

from spectral_connectivity import Multitaper, Connectivity

def multitaper_psd(time_series, fs, thbp=None, start_time=0.0, verbose=True):
    if thbp is not None:
        multitaper = Multitaper(
                time_series,
                sampling_frequency=fs,
                time_halfbandwidth_product=thbp,
                start_time=start_time
        )
    else:
        multitaper = Multitaper(
                time_series,
                sampling_frequency=fs,
                start_time=start_time
        )
    if verbose:
        print(f"Multitaper frequency resolution: {multitaper.frequency_resolution}")
        print(f"Multitaper number of tapers: {multitaper.n_tapers}")
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

def spectrum_trunc(freqs, spectrum, freq_range, spectrogram=False):
    """
    'Filters' a power spectra
    """
    freq_min,freq_max = freq_range
    trunc_idx = np.where(
        (freqs >= freq_min) & (freqs <= freq_max)
    )
    freqs_trunc = freqs[trunc_idx]
    if spectrogram:
        spectrum_trunc = spectrum[:,trunc_idx]
    else:
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

def plot_spectrum(x, fs, time_halfbandwidth_product=4, n_tapers=7, start_time=0.0, axes='loglog'):
    mt_psd, mt_freqs = multitaper_psd(x, fs, n_tapers, start_time)

    plt.figure(dpi=300)
    if axes=='loglog':
        plt.loglog(mt_freqs, mt_psd, linewidth=0.6, color='k')
    else:
        plt.plot(mt_freqs, mt_psd, linewidth=0.6, color='k')
    #plt.xlim((0,2000))

def plot_spectrogram(x, fs, spk_train=None, time_halfbandwidth_product=4, window_duration=0.05, window_step=0.025, start_time=0.0, ymin=0, ymax=6000):
    fig, ax = plt.subplots(dpi=300)
    connectivity = multitaper_spectrogram(x, fs, time_halfbandwidth_product, window_duration, window_step, start_time=start_time)
    im = ax.pcolormesh(
        connectivity.time,
        connectivity.frequencies,
        10 * np.log10(connectivity.power().squeeze().T),
        cmap="viridis",
        shading="auto"
        # vmin=-5.0,
        # vmax=-0.3
    )

    ax.set_ylim((ymin,ymax))
    #ax.set_yscale("log")

    
    if spk_train is not None:
        plt.scatter((spk_train / fs) + start_time, np.repeat(2, len(spk_train)), color='red', marker='^', s=3)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Power (dB/Hz)")