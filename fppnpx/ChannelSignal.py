import os
import numpy as np
import pandas as pd
import mat73 as mt
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt

from pathlib import Path
from tkinter import Tk, filedialog

# script for reading in SpikeGLX data from Jennifer Colonell (https://github.com/jenniferColonell/SpikeGLX_Datafile_Tools)
from .DemoReadSGLXData.readSGLX import readMeta, SampRate

from .preprocessing import *
from .spectral import *
from .filters import *
from .plottools import MATMAP

class ChannelSignal:
    """
    Object containing the time series, multitaper spectrum, and units
    """
    def __init__(self, channel, session_dataset, bandtype='AP', waveform_dataset=None,
                 notch_filt=3, bandpass_filt=None, highpass_filt=None,
                 compute_spectrum=True, time_halfbandwidth_product=None, compute_spectrogram=False, spectrogram_args=None,
                 verbose=True):
        bandtype = bandtype.lower()
        if (bandtype != 'lfp') and (bandtype != 'ap'):
            raise ValueError("Please specify band type (AP or LFP)")
        self.bandtype = bandtype
        print(f"Loading {bandtype.upper()} signal from channel {channel}...")
        
        fs = session_dataset[f"{bandtype}_sampling_rate"]  
        cluster_info = session_dataset["cluster_info"]        
        self.channel = channel
        self.sampling_rate = fs

        if waveform_dataset is not None:
            if verbose: print("Selecting good units...")
            self.dominant_units = np.intersect1d(cluster_info['cluster_id'][cluster_info['ch'] == channel], list(waveform_dataset.keys()))
            self.detected_units = np.intersect1d(session_dataset["clusters_on_channels"][channel], list(waveform_dataset.keys()))
        else:
            self.dominant_units = cluster_info['cluster_id'][cluster_info['ch'] == channel]
            self.detected_units = session_dataset["clusters_on_channels"][channel]
        
        # read in time-series signal and center
        if verbose: print("Reading in time-series...")
        channel_time_series,time_axis = read_bin(session_dataset[f"{bandtype}_bin_path"], session_dataset["time_window"], fs, session_dataset["N_channels"])
        time_series = channel_time_series[channel] - np.mean(channel_time_series[channel])
        self.time_axis = time_axis

        # filters, can change if needed
        if type(notch_filt) is int:
            if verbose: print("Notch filtering...")
            for nf in range(1, notch_filt+1):
                sos = sig.butter(1, [60*nf - 1, 60*nf + 1], 'bandstop', fs=fs, output='sos')
                time_series = sig.sosfilt(sos, time_series)

        if bandpass_filt is not None:
            if verbose: print(f"Band-pass filtering beteen {bandpass_filt[0]} and {bandpass_filt[1]} Hz...")
            sos = sig.butter(3, bandpass_filt, 'bp', fs=fs, output='sos')
            time_series = sig.sosfilt(sos, time_series)

        if highpass_filt is not None:
            if verbose: print(f"High-pass filtering at {highpass_filt} Hz...")
            sos = sig.butter(3, highpass_filt, 'hp', fs=fs, output='sos')
            time_series = sig.sosfilt(sos, time_series)
        self.time_series = time_series

        if compute_spectrum:
            # calculate multitaper spectrum
            if verbose: print("Computing multitapered spectrum...")
            self.mtap_spectrum, self.mtap_frequencies = multitaper_spectrum(time_series, fs, time_halfbandwidth_product=time_halfbandwidth_product, start_time=session_dataset['time_window'][0])

        if compute_spectrogram:
            if verbose: print("Computing multitapered spectrogram...")
            if spectrogram_args == None:
                spectrogram_args = {'window_duration': 0.5, 'window_step':0.5}
            self.mtap_spectrogram = multitaper_spectrogram(time_series, fs, time_halfbandwidth_product=time_halfbandwidth_product, start_time=session_dataset["time_window"][0], **spectrogram_args)

    def plot_time_series(self, display_clusters=None, ax=None):
        ax.plot(self.time_axis, self.time_series, color='k', linewidth=0.7)

    def plot_spectrum(self, display_clusters=None, log=False, ax=None):
        if log:
            ax.loglog(self.mtap_frequencies, self.mtap_spectrum, linewidth=0.7, color='k')
        else:
            ax.plot(self.mtap_frequencies, self.mtap_spectrum, linewidth=0.7, color='k')

        if self.bandtype == 'ap':
            ax.set_xlim(300,10000)
        elif self.bandtype == 'lfp':
            ax.set_xlim(0.5,500)

    def plot_spectrogram(self, vmin=-62, ax=None):
        if self.bandtype == 'lfp':
            ylim = (0.5, 500)
        elif self.bandtype == 'ap':
            ylim = (300,10000)
        im = ax.pcolormesh(
            self.mtap_spectrogram.time,
            self.mtap_spectrogram.frequencies,
            10 * np.log10(self.mtap_spectrogram.power().squeeze().T + 1e-12),
            cmap=MATMAP,
            shading='auto',
            vmin=vmin
        )
        ax.set_ylim(ylim)
        return im


        
