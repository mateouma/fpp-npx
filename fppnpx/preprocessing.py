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

from .spectral import *
from .filters import *
from .plottools import MATMAP

def read_bin(bin_path, time_window, fs, N_channels):
    bytes_per_sample = 2
    t1, t2 = int(np.round(time_window[0]*fs)), int(np.round(time_window[1]*fs))

    # read in Neuropixels signal
    with open(str(bin_path), 'rb') as f_src:
        # each sample for each channel is encoded on 16 bits = 2 bytes: samples*Nchannels*2.
        byte1 = int(t1*N_channels*bytes_per_sample)
        byte2 = int(t2*N_channels*bytes_per_sample)
        bytesRange = byte2-byte1

        f_src.seek(byte1)

        bData = f_src.read(bytesRange)
    
    signal_array = np.frombuffer(bData, dtype=np.int16)
    signal_array = signal_array.reshape((int(t2-t1), N_channels)).T

    time_axis = np.linspace(int(t1/fs), int(t2/fs), t2-t1)

    return signal_array, time_axis

def read_session(time_window, imec_path=None, amp_thres=0):
    """
    Initialize using the imec folder path
    """
    
    output = {}

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
    
    print(f"Loading session from {imec_path}")

    imec_dir = Path(imec_path)
    
    for path_cand in imec_dir.glob('*'):
        if path_cand.is_file():
            if 'imec0.ap.bin' in str(path_cand):
                ap_bin_path = path_cand
            elif 'imec0.lf.bin' in str(path_cand):
                lfp_bin_path = path_cand
        elif path_cand.is_dir():
            if 'kilosort4' in str(path_cand):
                kilosort_path = str(path_cand)

    output['ap_bin_path'] = ap_bin_path
    output['lfp_bin_path'] = lfp_bin_path

    print("Reading metadata...")
    ap_meta = readMeta(ap_bin_path)
    lfp_meta = readMeta(lfp_bin_path)
    output['ap_meta'] = ap_meta
    output['lfp_meta'] = lfp_meta

    fsAP = int(SampRate(ap_meta))
    fsLFP = int(SampRate(lfp_meta))
    output['ap_sampling_rate'] = fsAP
    output['lfp_sampling_rate'] = fsLFP

    N_channels = int(ap_meta['nSavedChans'])
    output['N_channels'] = N_channels

    # ========================================
    # READ IN ACTUAL CHANNEL TIME-SERIES ARRAY
    # ========================================
    print("Reading channel information...")
    channel_map = np.load(kilosort_path + "/channel_map.npy")
    output['channel_map'] = channel_map # TAKE NOTE OF THIS
    N_channels_inuse = channel_map.size
    output['N_channels_inuse'] = N_channels_inuse
    print(f"{N_channels_inuse} usable channels detected. Use `channel_map` to view usable channels.")

    t1, t2 = int(np.round(time_window[0]*fsAP)), int(np.round(time_window[1]*fsAP)) # for indexing spike times
    output['time_window'] = time_window

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
        cluster_spike_times[clust_id] = clust_spike_times[np.logical_and(clust_spike_times >= t1, clust_spike_times <= t2)]

        channels_for_clusterOI = np.where(template_amps[clust_to_template[clust_id],:] > amp_thres)[0]
        channels_for_clusters[clust_id] = channels_for_clusterOI

    output['cluster_spike_times'] = cluster_spike_times
    output['channels_for_clusters'] = channels_for_clusters

    print("Session loading complete.")
    return output

def load_waveforms(waveform_path=None, session_dataset=None, filter_truncs=None, default_trunc_idx=62):
    """
    Function for loading in the waveforms that will be used as filters. These come in from a custom MATLAB script from the Chand Lab using
    code from Daniel O'Shea. This can (and should) be updated later so that it can load the waveforms without the custom scripts.

    Returns:
        output (dict): dictionary containing the waveform mean, samples, channels, spike times, and filters for each cluster unit
        waveform_means (np.ndarray): N_units x N_timepoints array containing the waveform means (meant for WaveMAP)
    """
    print("Loading waveforms...")
    
    if waveform_path == None:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        waveform_path = filedialog.askopenfilename(title="Please select waveform .mat file")
        
        root.destroy()

    waveformmat = mt.loadmat(waveform_path)['goodUnits']
    fs = session_dataset['ap_sampling_rate']
    segment_length = session_dataset["time_window"][1] - session_dataset["time_window"][0]
    selected_clusters = waveformmat['cluId']
    print(f"{len(selected_clusters)} clusters with good waveforms detected.")

    output = {int(selected_cluster):{} for selected_cluster in selected_clusters}

    filter_trunc_dict = {int(selected_cluster):default_trunc_idx for selected_cluster in selected_clusters}
    try:
        for k in filter_truncs.keys():
            filter_trunc_dict[k] = filter_truncs[k]
    except:
        pass
    
    for i,selected_cluster in enumerate(selected_clusters):
        sel_clust_int = int(selected_cluster)
        waveform_samples = waveformmat['waveforms'][i]
        mean_waveform = np.mean(waveform_samples, axis=1)

        # calculate filters
        time_filter,freq_filter,filter_spectrum,filter_freq_axis = gen_filter(mean_waveform, n=fs, fs=fs, truncate_idx=filter_trunc_dict[sel_clust_int], center=True)
        
        output[sel_clust_int]['time_filter'] = time_filter
        output[sel_clust_int]['freq_filter'] = freq_filter
        output[sel_clust_int]['filter_spectrum'] = filter_spectrum
        output[sel_clust_int]['filter_freq_axis'] = filter_freq_axis
        output[sel_clust_int]['mean_waveform'] = mean_waveform
        output[sel_clust_int]['samples'] = waveform_samples
        output[sel_clust_int]['main_channel'] = waveformmat['channelId'][i]
        output[sel_clust_int]['channels'] = session_dataset['channels_for_clusters'][sel_clust_int]
        output[sel_clust_int]['spike_times'] = session_dataset['cluster_spike_times'][sel_clust_int]
        output[sel_clust_int]['firing_rate'] = len(session_dataset['cluster_spike_times'][sel_clust_int]) / segment_length

    waveform_means = np.mean(waveformmat['waveforms'], axis=2)
    return output, waveform_means
