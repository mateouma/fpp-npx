import numpy as np
import matplotlib.pyplot as plt

import fppnpx as fn
from fppnpx.ChannelSignal import ChannelSignal
from fppnpx.FPPGLM import FPPGLM
from fppnpx.signalfuncs import load_signal, gen_all_channel_signals
from wavemapnpx.WaveMAPClassifier import WaveMAPClassifier

from plottools import WAVEMAP_PAL, ml_map, cmap

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

apath = "/Users/mateouma/Downloads/monkey datasets/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0_t0.exported.imec0.ap-001.bin"
cpath = "/Users/mateouma/Downloads/monkey datasets/cluster_info_task.tsv"
wpath = "/Users/mateouma/Downloads/monkey datasets/20230630_DLPFCwaveforms.mat"

fs = 30000
time_window = [204,209] # seconds within the recording # [100,123]

signal_dataset = load_signal(appath=apath, time_window=time_window, fs=fs, cipath=cpath, wfpath=wpath)

channel_list = [163, 208, 205]

channel_signals = {}
for chan in channel_list:
    channel_signals[f"ch{chan}"] = ChannelSignal(channel=chan, signal_dataset=signal_dataset, hpf=300, high_pass_filt=False)

# generate muiltitapered spectra
ch163_PSD,mt_freqs = fn.spectrafuncs.multitaper_psd(channel_signals['ch163'].time_series, fs=fs, thbp=5.0, start_time=channel_signals['ch163'].time_axis)
ch208_PSD,mt_freqs = fn.spectrafuncs.multitaper_psd(channel_signals['ch208'].time_series, fs=fs, thbp=5.0, start_time=channel_signals['ch163'].time_axis)

u296_instances,u296_filters,theor_freqs = channel_signals['ch163'].generate_unit_filters(selected_unit=296, truncate_idx=74)
u378_instances,u378_filters,_ = channel_signals['ch208'].generate_unit_filters(selected_unit=378)
u379_instances,u379_filters,_ = channel_signals['ch208'].generate_unit_filters(selected_unit=379)
u374_instances,u374_filters,_ = channel_signals['ch205'].generate_unit_filters(selected_unit=374)
u375_instances,u375_filters,_ = channel_signals['ch205'].generate_unit_filters(selected_unit=375)

def plot_chan_sig(channel_signal):
    fig, ax = plt.subplots()
    

fig, ax = plt.subplots(1,3, figsize=(8,3))

ax[0].plot(channel_signals['ch163'].time_axis, channel_signals['ch163'].time_series, color='k', linewidth=0.2, zorder=0)
for k,i in enumerate(channel_signals['ch163'].units):
    ax[0].scatter((channel_signals['ch163'].spike_times[i] + channel_signals['ch163'].t1)/fs, np.repeat(-30, len(channel_signals['ch163'].spike_times[i])), zorder=2, marker='^', s=4, alpha=0.7, label=f'unit {i}', color=cmap(k))
ax[0].legend(fontsize='x-small')

ax[1].loglog(mt_freqs, ch163_PSD, color='k', linewidth=0.6)
ax[1].loglog(theor_freqs, u296_filters['waveform_instance']['filter_psd'], color=cmap(0), linestyle='--', linewidth=0.8, label='Waveform PSD')
ax[1].loglog(theor_freqs, u296_filters['waveform_instance']['filter_psd'] * channel_signals['ch163'].firing_rates[296], color=cmap(0), label='Theoretical PSD (true $\lambda_0$)')
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Log Power")
ax[1].set_xlim((1,6000))

ax[1].spines['bottom'].set_bounds(1, 6e3)
ax[1].spines['left'].set_bounds(2e-6, 5.0)
ax[1].legend(fontsize='x-small')

ax[2].set_title("QQ-plot")

fig.tight_layout()

fig.savefig("figs/temp_ts.svg", dpi=300)


plt.show(block=True)