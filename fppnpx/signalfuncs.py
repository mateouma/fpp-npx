import numpy as np
import pandas as pd
import mat73 as mt
import matplotlib.pyplot as plt

from .ChannelSignal import ChannelSignal

def load_signal(appath, time_window, fs, cipath, wfpath):
    """
    appath (str): file path for AP band
    time_window (arr): range of time in seconds, ex. [0,5]
    fs (int): sampling rate
    cipath (str): cluster info (unit classification and channel loc) file path
    wfpath (str): waveform info (from kilosort/phy) file path

    Returns:
    
    """
    Nchans = 385
    bytes_per_sample = 2

    t1, t2 = int(np.round(time_window[0]*fs)), int(np.round(time_window[1]*fs))

    # read in Neuropixels signal
    with open(appath, 'rb') as f_src:
        # each sample for each channel is encoded on 16 bits = 2 bytes: samples*Nchannels*2.
        byte1 = int(t1*Nchans*bytes_per_sample)
        byte2 = int(t2*Nchans*bytes_per_sample)
        bytesRange = byte2-byte1

        f_src.seek(byte1)

        bData = f_src.read(bytesRange)
    
    rc = np.frombuffer(bData, dtype=np.int16)
    rc = rc.reshape((int(t2-t1), Nchans)).T
    rc = rc[:-1,:] # reshape into N X (fs x time) shape array

    # read in cluster info
    ci = pd.read_csv(cipath, sep="\t")
    ci = ci[ci['group'] == 'good']

    # read in waveform data
    wf = mt.loadmat(wfpath)['goodUnits']
    wf_pd = pd.DataFrame(mt.loadmat(wfpath)['goodUnits'])

    chan_unit_dict = {}

    all_channels = np.array(wf_pd['channelId'], dtype='int')
    all_units = np.array(wf_pd['cluId'], dtype='int')

    for chan in np.unique(all_channels):
        chan_unit_dict[chan] = []

    for i,unit in enumerate(all_units):
        chan_unit_dict[all_channels[i]].append(unit)

    # for chan in np.unique(all_channels):
    #     chan_unit_dict[int(chan)] = []

    # for i in range(len(all_units)):
    #     chan_unit_dict[int(all_channels[i])].append(int(all_units[i]))

    signal_dataset = {
        "time_series": rc,
        "cluster_info": ci,
        "waveform_info": wf,
        "time_range": (t1, t2),
        "sampling_rate": fs,
        "channel_unit_index": chan_unit_dict,
        "channels": all_channels,
        "units": all_units
    }

    return signal_dataset

def gen_all_channel_signals(signal_dataset, hpf=300, filt=False):
    channel_signals = {}
    for chan in np.unique(signal_dataset["channels"]):
        channel_signals[f"ch{chan}"] = ChannelSignal(channel=chan, signal_dataset=signal_dataset, hpf=hpf, filt=filt)
    print(f"Generated {len(channel_signals.keys())} channel signals.")
    return channel_signals