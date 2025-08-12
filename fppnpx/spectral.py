import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

from spectral_connectivity import Multitaper, Connectivity

def multitaper_spectrum(time_series, fs, time_halfbandwidth_product=None, start_time=0.0, verbose=False):
    if time_halfbandwidth_product is not None:
        multitaper = Multitaper(
                time_series,
                sampling_frequency=fs,
                time_halfbandwidth_product=time_halfbandwidth_product,
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
    multitaper_spectrum = connectivity.power().squeeze()
    multitaper_frequencies = connectivity.frequencies

    return multitaper_spectrum, multitaper_frequencies

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
    Truncates a power spectrum within a certain frequency range
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
    Interpolates a power spectrum
    """
    spectrum_interpolated = np.interp(
        reference_freqs,
        data_freqs,
        spectrum
    )

    return spectrum_interpolated