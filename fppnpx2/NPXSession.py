import os
import numpy as np
import matplotlib.pyplot
import pandas as pd
import mat73 as mt
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt

from pathlib import Path
from tkinter import Tk, filedialog

# script for reading in SpikeGLX data from Jennifer Colonell (https://github.com/jenniferColonell/SpikeGLX_Datafile_Tools)
from .DemoReadSGLXData.readSGLX import readMeta, SampRate
from .spectral import *
from .filters import *

def read_session(time_window, imec_path=None, bandtype='', amp_thres=0, waveform_path=None):
    """
    Initialize using the imec folder path
    """
    # /home/mateoumaguing/Documents/MiscData/TibsCFD/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0
    # /home/mateoumaguing/Documents/MiscData/TibsCFD/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0_t0.exported.imec0.ap.bin

    output = {}

    bandtype = bandtype.lower()
    if (bandtype != 'lfp') and (bandtype != 'ap'):
        raise ValueError("Please specify band type (AP or LFP)")
    output['bandtype'] = bandtype

    # =====================================
    # READ IN FILE FOLDER PATH AND METADATA
    # =====================================
    if imec_path == None:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        # get imec0 folder that contains .bin files and kilosort4 folder
        imec_path = filedialog.askdirectory(title="Please select imec0 folder")
        output['imec_path'] = imec_path
        
        root.destroy()
    
    print(f"Creating session from {imec_path}")

    # read in metadata based on AP or LFP binary file
    if bandtype == 'ap':
        search_string = 'imec0.ap.bin'
    elif bandtype == 'lfp':
        search_string = 'imec0.lf.bin'
    imec_dir = Path(imec_path)
    
    for path_cand in imec_dir.glob('*'):
        if path_cand.is_file():
            if search_string in str(path_cand):
                bin_path = path_cand
        elif path_cand.is_dir():
            if 'kilosort4' in str(path_cand):
                kilosort_path = str(path_cand)
    
    print("Reading metadata...")
    meta = readMeta(bin_path)
    output['meta'] = meta

    fs = int(SampRate(meta))
    output['sampling_rate'] = fs

    N_channels = int(meta['nSavedChans'])
    output['N_channels'] = N_channels

    # ========================================
    # READ IN ACTUAL CHANNEL TIME-SERIES ARRAY
    # ========================================
    print("Reading recorded signal...")
    channel_map = np.load(kilosort_path + "/channel_map.npy")
    output['channel_map'] = channel_map # TAKE NOTE OF THIS
    N_channels_inuse = channel_map.size
    output['N_channels_inuse'] = N_channels_inuse
    print(f"{N_channels_inuse} usable channels. Use `channel_map` to get usable channels.")

    bytes_per_sample = 2
    t1, t2 = int(np.round(time_window[0]*fs)), int(np.round(time_window[1]*fs))
    output['time_window'] = time_window
    output['time_range'] = (t1,t2)

    # read in Neuropixels signal
    with open(str(bin_path), 'rb') as f_src:
        # each sample for each channel is encoded on 16 bits = 2 bytes: samples*Nchannels*2.
        byte1 = int(t1*N_channels*bytes_per_sample)
        byte2 = int(t2*N_channels*bytes_per_sample)
        bytesRange = byte2-byte1

        f_src.seek(byte1)

        bData = f_src.read(bytesRange)
    
    channel_ts_array = np.frombuffer(bData, dtype=np.int16)
    channel_ts_array = channel_ts_array.reshape((int(t2-t1), N_channels)).T
    # channel_ts_array = channel_ts_array[channel_map,:] # index by usable channels into into N_channels_inuse X (fs x time) shape array
    output['channel_ts_array'] = channel_ts_array

    # ==================================
    # READ IN CLUSTER INFO FROM KILOSORT
    # ==================================
    print("Reading cluster information from kilosort...")
    cluster_info = pd.read_csv(kilosort_path + "/cluster_info.tsv", sep="\t")
    cluster_info = cluster_info[cluster_info["group"] == "good"] # only select good units
    output['cluster_info'] = cluster_info

    templates = np.load(kilosort_path + "/templates.npy") # should be a N_all_units x N_waveform_timepoints (61) x N_channels_inuse array
    spike_templates = np.load(kilosort_path + "/spike_templates.npy") # for each spike within the recording, which template was matched
    spike_times = np.load(kilosort_path + "/spike_times.npy")
    spike_clusters = np.load(kilosort_path + "/spike_clusters.npy") # for each spike within the recording, which cluster was assigned

    template_to_clust = {t:c for t,c in zip(spike_templates, spike_clusters)}
    clust_to_template = {c:t for c,t in zip(spike_clusters, spike_templates)}
    template_amps = np.max(np.abs(templates), axis=1) # N_templates x channels array containing maximum amplitude of each template on each channel
    clusters_on_channels = {}
    for i,chan in enumerate(channel_map):
        templates_on_chanOI = np.where(template_amps[:,i] > amp_thres)[0] # which templates greater than `amp_thres` were detected on channel `chan`
        clusters_on_channels[chan] = np.array([template_to_clust[toc] for toc in templates_on_chanOI])
    output["clusters_on_channels"] = clusters_on_channels

    channels_for_clusters = {}
    cluster_spike_times = {}
    for i in range(len(cluster_info)):
        clust_id,clust_ch = cluster_info.iloc[i][['cluster_id', 'ch']]

        clust_spike_times = spike_times[spike_clusters == clust_id]
        cluster_spike_times[clust_id] = clust_spike_times

        channels_for_clusterOI = np.where(template_amps[clust_to_template[clust_id],:] > amp_thres)[0]
        channels_for_clusters[clust_id] = channels_for_clusterOI

    output['cluster_spike_times'] = cluster_spike_times
    output['channels_for_clusters'] = channels_for_clusters

    print("Session loading complete.")
    return output

def load_waveforms(waveform_path, session_dataset, filter_truncs=None):
    output = {}

    print("Loading waveforms...")
    waveform_samples = sio.loadmat(waveform_path)
    fs = session_dataset['sampling_rate']
    selected_clusters = np.squeeze(waveform_samples['cluster_id'])

    print(f"{selected_clusters.size} waveforms selected.")
    output['cluster_id'] = selected_clusters
    output['samples'] = waveform_samples['waveform_sample']
    output['waveform_means'] = waveform_samples['waveform_sample'].mean(axis=2)
    output['spike_times'] = [session_dataset['cluster_spike_times'][wcl] for wcl in selected_clusters]
    output['channels'] = [session_dataset['channels_for_clusters'][wcl] for wcl in selected_clusters]

    if filter_truncs == None:
        filter_truncs = {}

    cluster_time_filters = []
    cluster_freq_filters = []
    cluster_filter_psds = []

    for i,wcl in enumerate(selected_clusters):
        if filter_truncs.get(wcl) == None:
            ftrunc = 62
        else:
            ftrunc = filter_truncs.get(wcl)
        time_filter,freq_filter,filter_psd,filter_freq_axis = gen_filter(output['waveform_means'][i], fs=fs, truncate_idx=ftrunc, center=True)
        cluster_time_filters.append(time_filter)
        cluster_freq_filters.append(freq_filter)
        cluster_filter_psds.append(filter_psd)


    return output


class ChannelSignal:
    """
    Object containing the time series and units
    """
    def __init__(self, channel, session_dataset, waveform_dataset, high_pass_filt=None, multitaper_args=None):
        """
        Initialize
        """
        fs = session_dataset["sampling_rate"]
        channel_time_series = session_dataset["channel_ts_array"]
        cluster_info = session_dataset["cluster_info"]
        time_range = session_dataset["time_range"]
        
        self.channel = channel
        self.fs = fs
        self.dominant_units = np.intersect1d(cluster_info['cluster_id'][cluster_info['ch'] == channel], waveform_dataset['cluster_id'])
        self.detected_units = np.intersect1d(session_dataset["clusters_on_channels"][channel], waveform_dataset['cluster_id'])

        self.time_axis = np.linspace(int(time_range[0]/fs), int(time_range[1]/fs), time_range[1] - time_range[0])

        # center
        time_series = channel_time_series[channel] - np.mean(channel_time_series[channel])

        # filter, can change if needed
        sos = sig.butter(1, [59,61], 'bandstop', fs=fs, output='sos')
        time_series = sig.sosfilt(sos, time_series)

        sos = sig.butter(1, [119,121], 'bandstop', fs=fs, output='sos')
        time_series = sig.sosfilt(sos, time_series)

        sos = sig.butter(1, [179,181], 'bandstop', fs=fs, output='sos')
        time_series = sig.sosfilt(sos, time_series)

        if high_pass_filt is not None:
            sos = sig.butter(3, high_pass_filt, 'hp', fs=fs, output='sos')
            time_series = sig.sosfilt(sos, time_series)
        self.time_series = time_series

        # calculate multitaper psd
        mtap_psd, mtap_frequencies = multitaper_psd(time_series, fs, start_time=session_dataset['time_window'][0])
        self.mtap_psd = mtap_psd
        self.mtap_frequencies = mtap_frequencies

    def plot_time_series(self, display_clusters=None, ax=None):
        ax.plot(self.time_axis, self.time_series, color='k')

    def plot_spectrum(self, display_clusters=None, log=False, ax=None):
        if log:
            ax.loglog(self.mtap_frequencies, self.mtap_psd, linewidth=0.7, color='k')
        else:
            ax.plot(self.mtap_frequencies, self.mtap_psd, linewidth=0.7, color='k')


        
