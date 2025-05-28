import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

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

channel_list = [208, 205, 209, 212] # [163] # 

channel_signals = {}
for chan in channel_list:
    channel_signals[f"ch{chan}"] = ChannelSignal(channel=chan, signal_dataset=signal_dataset, hpf=300, high_pass_filt=False)

# u296_instances,u296_filters,theor_freqs = channel_signals['ch163'].generate_unit_filters(selected_unit=296, truncate_idx=74)
u378_instances,u378_filters,theor_freqs = channel_signals['ch208'].generate_unit_filters(selected_unit=378)
u379_instances,u379_filters,_ = channel_signals['ch208'].generate_unit_filters(selected_unit=379)
u374_instances,u374_filters,_ = channel_signals['ch205'].generate_unit_filters(selected_unit=374)
u375_instances,u375_filters,_ = channel_signals['ch205'].generate_unit_filters(selected_unit=375)
u380_instances,u380_filters,_ = channel_signals['ch209'].generate_unit_filters(selected_unit=380)
u383_instances,u383_filters,_ = channel_signals['ch212'].generate_unit_filters(selected_unit=383)

# make action potential waveform * spike train

# orig_ts_signal = channel_signals['ch163'].time_series

# ts_signal = np.zeros(fs*(time_window[1]-time_window[0]))

# for i,spk_time in enumerate(channel_signals['ch163'].spike_times[296].astype(int)):
#     ts_signal[(spk_time-20):(spk_time+53)] = orig_ts_signal[(spk_time-20):(spk_time+53)]

orig_ts_signal = channel_signals['ch208'].time_series

ts_signal = np.zeros(fs*(time_window[1]-time_window[0]))

for i,spk_time in enumerate(channel_signals['ch208'].spike_times[378].astype(int)):
    ts_signal[(spk_time-20):(spk_time+42)] = orig_ts_signal[(spk_time-20):(spk_time+42)]

for i,spk_time in enumerate(channel_signals['ch208'].spike_times[379].astype(int)):
    ts_signal[(spk_time-20):(spk_time+42)] = orig_ts_signal[(spk_time-20):(spk_time+42)]

for i,spk_time in enumerate(channel_signals['ch205'].spike_times[374].astype(int)):
    ts_signal[(spk_time-20):(spk_time+42)] = orig_ts_signal[(spk_time-20):(spk_time+42)]

for i,spk_time in enumerate(channel_signals['ch205'].spike_times[375].astype(int)):
    ts_signal[(spk_time-20):(spk_time+42)] = orig_ts_signal[(spk_time-20):(spk_time+42)]

for i,spk_time in enumerate(channel_signals['ch209'].spike_times[380].astype(int)):
    ts_signal[(spk_time-20):(spk_time+42)] = orig_ts_signal[(spk_time-20):(spk_time+42)]

for i,spk_time in enumerate(channel_signals['ch212'].spike_times[383].astype(int)):
    ts_signal[(spk_time-20):(spk_time+42)] = orig_ts_signal[(spk_time-20):(spk_time+42)]


# ts_signal = channel_signals['ch208'].time_series

# multitaper PSD

freq_range = (1,6000)

# filter_psds = np.array([u296_filters['waveform_instance']['filter_psd']])
# sel_firing_rates = np.array([channel_signals['ch163'].firing_rates[296]])

filter_psds = np.array([u378_filters['waveform_instance']['filter_psd_iaw'],
                        u379_filters['waveform_instance']['filter_psd_iaw'],
                        u374_filters['waveform_instance']['filter_psd_iaw'],
                        u375_filters['waveform_instance']['filter_psd_iaw'],
                        u380_filters['waveform_instance']['filter_psd_iaw'],
                        u383_filters['waveform_instance']['filter_psd_iaw']])
sel_firing_rates = np.array([channel_signals['ch208'].firing_rates[378],
                             channel_signals['ch208'].firing_rates[379],
                             channel_signals['ch205'].firing_rates[374],
                             channel_signals['ch205'].firing_rates[375],
                             channel_signals['ch209'].firing_rates[380],
                             channel_signals['ch212'].firing_rates[383]])

mt_PSD,mt_freqs = fn.spectrafuncs.multitaper_psd(ts_signal, fs=fs, start_time=time_window[0])

theor_spk_PSD_units = filter_psds * sel_firing_rates[:,None]
theor_spk_PSD = np.sum(theor_spk_PSD_units, axis=0)

# theor_spk_PSD = filter_psds[0] * sel_firing_rates[0]

# QQ plot
mt_PSD_trunc,mt_freqs_trunc = fn.spectrafuncs.spectrum_trunc(mt_freqs,mt_PSD,freq_range)
theor_spk_PSD_trunc,theor_freqs_trunc = fn.spectrafuncs.spectrum_trunc(theor_freqs,theor_spk_PSD,freq_range)
theor_spk_PSD_interp = fn.spectrafuncs.spectrum_interp(mt_freqs_trunc, theor_freqs_trunc, theor_spk_PSD_trunc)

Z_f = np.sort((5 * mt_PSD_trunc) / theor_spk_PSD_interp)
n = Z_f.size

theor_quant = stats.gamma.ppf((np.arange(1, n + 1) - 0.5) / n, a=5)

fig, ax = plt.subplots(1,3, figsize=(10,3))

ax = ax.flatten()

ax[0].plot(channel_signals['ch208'].time_axis, ts_signal, color='k', linewidth=0.6)
# ax[0].plot(channel_signals['ch163'].time_axis, ts_signal, color='k', linewidth=0.6)
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Voltage (mV)")
ax[0].set_xlim((203.5, 209.5))
ax[0].set_ylim((-150, 150))
ax[0].spines['bottom'].set_bounds(204, 209)
ax[0].spines['left'].set_bounds(-150, 150)

ax[1].loglog(mt_freqs, mt_PSD, color='k', linewidth=0.6)
# ax[1].loglog(theor_freqs, theor_spk_PSD,
#              color=cmap(1), label='Theoretical PSD ($\lambda_0\cdot$ AP PSD)')
ax[1].loglog(theor_freqs, theor_spk_PSD_units[0],linewidth=0.8,
             color=cmap(1), label='$\lambda_0^1\cdot$ unit 1 PSD')
ax[1].loglog(theor_freqs, theor_spk_PSD_units[1],linewidth=0.8,
             color=cmap(2), label='$\lambda_0^2\cdot$ unit 2 PSD')
ax[1].loglog(theor_freqs, theor_spk_PSD_units[2],linewidth=0.8,
             color=cmap(3), label='$\lambda_0^3\cdot$ unit 3 PSD')
ax[1].loglog(theor_freqs, theor_spk_PSD_units[3],linewidth=0.8,
             color=cmap(4), label='$\lambda_0^4\cdot$ unit 4 PSD')
ax[1].loglog(theor_freqs, theor_spk_PSD_units[4],linewidth=0.8,
             color=cmap(5), label='$\lambda_0^4\cdot$ unit 4 PSD')
ax[1].loglog(theor_freqs, theor_spk_PSD_units[5],linewidth=0.8,
             color=cmap(6), label='$\lambda_0^4\cdot$ unit 4 PSD')
ax[1].loglog(theor_freqs, theor_spk_PSD,
             color='darkgray', label='Theoretical PSD (sum)')
ax[1].legend(fontsize='x-small')
ax[1].set_xlim(freq_range)
ax[1].set_ylim((1.5e-5, 1))
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Log Power")
ax[1].spines['left'].set_bounds(2e-5, 1)
ax[1].spines['bottom'].set_bounds(1, 6e3)

ax[2].scatter(theor_quant, Z_f, marker='x', c='k', s=1)
ax[2].plot(np.arange(-100,100), np.arange(-100,100), color='red', linewidth=0.5)
ax[2].set_xlim((-1, 21))
ax[2].set_ylim((-1, 21))
ax[2].set_xlabel("Theoretical quantiles")
ax[2].set_ylabel("Sample quantiles")
ax[2].spines['bottom'].set_bounds(0, 20)
ax[2].spines['left'].set_bounds(0, 20)

fig.tight_layout()
# fig.savefig("figs/tri_sig_plot_4unitfull.svg", dpi=300)

plt.show(block=True)