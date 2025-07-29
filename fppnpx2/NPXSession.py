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

    # TO-DO: FIX THIS SO IT IS CLUSTERS ON EACH CHANNEL
    template_to_clust = {t:c for t,c in zip(spike_templates, spike_clusters)}
    template_amps = np.max(np.abs(templates), axis=1) # N_templates x channels array containing maximum amplitude of each template on each channel
    templates_on_channels = {}
    for i,chan in enumerate(channel_map):
        templates_on_chanOI = np.where(template_amps[:,i] > amp_thres)[0] # which templates greater than `amp_thres` were detected on channel `chan`
        templates_on_channels[chan] = templates_on_chanOI

    cluster_spike_times = {}
    for i in range(len(cluster_info)):
        clust_id,clust_ch = cluster_info.iloc[i][['cluster_id', 'ch']]

        clust_spike_times = spike_times[spike_clusters == clust_id]
        cluster_spike_times[clust_id] = clust_spike_times

    # for cluster in cluster_info['cluster_id']:
    #     cluster_spike_times[cluster] = spike_times[spike_clusters == cluster]
    output['cluster_spike_times'] = cluster_spike_times

    templates.close()
    spike_templates.close()
    spike_times.close()
    spike_clusters.close()

    if waveform_path is not None:
        # THE WAVEFORM .mat IS CUSTOM MADE, WILL NEED TO GENERATE FROM CHAND LAB FOR EACH NEW DATASET
        waveform_samples = sio.loadmat(waveform_path)
    output['waveform_samples'] = waveform_samples

    return output

class ChannelSignal:
    """
    Object containing the time series and units
    """
    def __init__(self, channel, session_dataset, high_pass_filt=None):
        """
        Initialize
        """
        fs = session_dataset["sampling_rate"]
        channel_time_series = session_dataset["channel_ts_array"]
        cluster_info = session_dataset["cluster_info"]
        waveforms = session_dataset["waveform_samples"]
        time_range = session_dataset["time_range"]
        
        self.channel = channel
        self.fs = fs
        self.dominant_units = cluster_info['cluster_id'][cluster_info['ch'] == channel]

        self.time_axis = np.linspace(int(time_range[0]/fs), int(time_range[1]/fs), time_range[1] - time_range[0])

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


        
