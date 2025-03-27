import numpy as np
from matplotlib import colormaps
import scipy.signal as sig
import matplotlib.pyplot as plt

from .filterfuncs import *

class ChannelSignal:
    """
    An object containing the time series, units, and respective waveforms for a given channel
    """

    def __init__(self, channel, signal_dataset, hpf=300, filt=False):
        """
        Initialize

        signal_dataset (dict): dictionary containing channel time series, sampling rate, cluster and waveform information, time range
        hpf (int): the frequency at which to high pass filter
        filt (bool): whether or not to filter the signal

        For each unit, the spike times, firing rates, and waveforms will be stored in dictionaries
        with the unit number as the key.
        """
        self.channel = channel

        fs = signal_dataset["sampling_rate"]
        rc = signal_dataset["time_series"]
        ci = signal_dataset["cluster_info"]
        wf = signal_dataset["waveform_info"]
        times = signal_dataset["time_range"]
        
        self.fs = fs

        goodchannels = np.array(wf['channelId']).astype(int)

        # array of the indices within 'wf' corresponding to the units on the channel
        self.unit_idx, = np.where(goodchannels == channel) 
        self.units = np.array(wf['cluId'])[self.unit_idx].astype(int)

        # spectrum spikes: 60, 120, 180, 300, 360, 420, 540, 660
        # center and filter time series
        time_series = rc[channel] - np.mean(rc[channel])
        b_notch, a_notch = sig.iirnotch(60.0, 30.0, fs)
        time_series = sig.filtfilt(b_notch, a_notch, time_series)
        if filt:
            sos = sig.butter(3, hpf, 'hp', fs=fs, output='sos')
            time_series = sig.sosfilt(sos, time_series)
        self.time_series = time_series

        # create time axis
        t1,t2 = times
        self.t1 = t1
        self.N = t2 - t1
        self.time_axis = np.linspace(int(t1/fs),int(t2/fs),self.N)

        # for each unit, find the spike times and calculate firing rates
        time_restrict = lambda st: st[np.logical_and(st >= t1, st <= t2)] - t1 # get spikes within selected time range
        self.spike_times = {self.units[i]: time_restrict(wf['st'][ui]) for i,ui in enumerate(self.unit_idx)}
        self.firing_rates = {self.units[i]: len(self.spike_times[ui])/(int(self.N/fs)) for i,ui in enumerate(self.units)}

        # designate the cluster and waveform information
        self.cluster_info = ci[ci['ch'] == channel]
        self.waveforms = {self.units[i]: wf['waveforms'][ui] for i,ui in enumerate(self.unit_idx)}
        self.waveform_clusters = {self.units[i]: wf['cluster_waveforms'][ui][:,0] for i,ui in enumerate(self.unit_idx)}
        

    def plot_signal(self, dpi=300, spky=-30, xlim=None, additional_spikes=None, add_spk_units=None):
        """
        Plots the time series with the spikes superimposed
        """
        cmap = colormaps.get_cmap('Set1')
        plt.figure(figsize=(8,3), dpi=dpi)
        plt.plot(self.time_axis, self.time_series, color='k', linewidth=0.2, zorder=0)
        for k,i in enumerate(self.units):
            plt.scatter((self.spike_times[i]+ self.t1)/self.fs, np.repeat(spky, len(self.spike_times[i])), 
                        zorder=2, marker='^', s=4, alpha=0.7, label=f'unit {i}', color=cmap(k))
        
        # for plotting spikes from units located on other channels
        if additional_spikes is not None:
            for j in range(len(additional_spikes)):
                plt.scatter((additional_spikes[j]+self.t1)/self.fs, np.repeat(spky-(2*(j+1)), len(additional_spikes[j])),
                            zorder=2, marker='^', s=4, alpha=0.7, label=f'unit {add_spk_units[j]}', color=cmap(k+j+1))
                
        if xlim is not None:
            plt.xlim(xlim)

        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.title(f"Channel {self.channel} Time-Series")

        plt.legend()
        
    def generate_unit_filters(self, selected_unit, truncate_idx=62, n=None, truncate=True):
        """
        Generate instance-averaged filters as well as filters for the average waveform
        """
        if n is None:
            n = self.fs

        filter_add_length = truncate_idx - 20
        
        # filters for average waveform
        ch_wf_mean = self.waveforms[selected_unit].mean(axis=1)
        filter_t, filter_f, filter_psd, theor_freqs = gen_filter(ch_wf_mean, n, self.fs, center=True)

        average_waveform_filters = {
            'time_filter': filter_t,
            'freq_filter': filter_f,
            'filter_psd': filter_psd
        }

        # filters for each instance of the waveform
        waveform_instances = []

        # for each event instance, find the waveform and calculate the spectrum
        spk_instances = 0
        for _,spktime in enumerate(self.spike_times[selected_unit]):
            if spktime < 20 or (spktime + 42) > self.N:
                continue # ignore spikes with waveform window outside of signal time range
            spk_instances += 1
            waveform_instance = self.time_series[int(spktime-20):int(spktime+filter_add_length)]
            waveform_instances.append(waveform_instance)

        # average across each instance of the waveform within the time series
        instance_avg_waveform = np.array(waveform_instances).mean(axis=0)
        _, __,filter_psd_iaw,___ = gen_filter(instance_avg_waveform, n, self.fs, truncate_idx=truncate_idx, center=True)

        time_filter_instances = []
        frequency_filter_instances = []
        filter_psd_instances = []
        for waveform_instance in waveform_instances:
            filter_ti,filter_fi,filter_psdi,_ = gen_filter(waveform_instance, n, self.fs, truncate_idx=truncate_idx, center=True)
            time_filter_instances.append(filter_ti)
            frequency_filter_instances.append(filter_fi)
            filter_psd_instances.append(filter_psdi)

        waveform_instance_filters = {
            'time_filter': np.array(time_filter_instances).mean(axis=0),
            'freq_filter': np.array(time_filter_instances).mean(axis=0),
            'filter_psd': np.array(filter_psd_instances).mean(axis=0),
            'filter_psd_iaw': filter_psd_iaw
        }

        instances = {
            "waveforms": waveform_instances,
            "psds": filter_psd_instances,
            "spikes": spk_instances
        }

        filters = {
            'average_waveform': average_waveform_filters,
            'waveform_instance': waveform_instance_filters
        }
        
        return instances, filters, theor_freqs
