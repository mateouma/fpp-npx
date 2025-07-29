import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from scipy import stats

import fppnpx as fn
from fppnpx.ChannelSignal import ChannelSignal
from fppnpx.FPPGLM import FPPGLM
from fppnpx.signalfuncs import load_signal, gen_all_channel_signals
from wavemapnpx.WaveMAPClassifier import WaveMAPClassifier

from plottools import WAVEMAP_PAL, MISC6_PAL, ml_map, cmap

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

apath = "/Users/mateouma/Downloads/monkey datasets/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0_t0.exported.imec0.ap-001.bin"
cpath = "/Users/mateouma/Downloads/monkey datasets/cluster_info_task.tsv"
wpath = "/Users/mateouma/Downloads/monkey datasets/20230630_DLPFCwaveforms.mat"

fs = 30000
time_window = [204,209] # seconds within the recording # [100,123]

signal_dataset = load_signal(appath=apath, time_window=time_window, fs=fs, cipath=cpath, wfpath=wpath)

# =========================================
# CONFIGURE PARAMS HERE
# =========================================

chan_unitOI_dict = {163: [296]}

# chan_unitOI_dict = { # take note of the order!
#     208: [378,379],
#     205: [374,375],
#     209: [380],
#     212: [383]
# }

truncate_idx = 74

# truncate_idx = 62

# ts_mode = 'spikes'

ts_mode = 'raw'

# ==========================================

channel_list = list(chan_unitOI_dict.keys())
channel_OI = channel_list[0]

filter_add_length = truncate_idx - 20

channel_signals = {}
for chan in channel_list:
    channel_signals[f"ch{chan}"] = ChannelSignal(channel=chan, signal_dataset=signal_dataset, hpf=300, high_pass_filt=False)

# select channel of interest

orig_channel_sig = channel_signals[f'ch{channel_OI}']
orig_channel_sig_ts = orig_channel_sig.time_series
orig_channel_sig_ta = orig_channel_sig.time_axis

# find instances of each unit within the original time series, calculate filters

unit_spike_ts_list = []
unit_spike_indicators = []
filter_psds = []
sel_firing_rates = []

for chID,unit_list in chan_unitOI_dict.items():
    channel_sig_temp = channel_signals[f'ch{chID}']
    u_counter = 1
    for unitOI in unit_list:
        orig_channel_uOI_instances = []
        unit_spike_indicator = np.zeros_like(orig_channel_sig_ts, dtype=int)
        for spktime in channel_sig_temp.spike_times[unitOI]:
            if spktime < 20 or (spktime + 42) > channel_sig_temp.N:
                continue
            unit_spike_indicator[int(spktime-20):int(spktime+filter_add_length)] = 1
            orig_channel_uOI_instances.append(orig_channel_sig_ts[int(spktime-20):int(spktime+filter_add_length)])
        # u_counter += 1
        unit_spike_indicators.append(unit_spike_indicator)
        orig_channel_unit_wf = np.mean(orig_channel_uOI_instances, axis=0)
        scale_to = np.max(np.abs(orig_channel_unit_wf))
        _,unit_filters,theor_freqs = channel_sig_temp.generate_unit_filters(selected_unit=unitOI, scale_to=scale_to)

        # spike trains
        unit_spike_train = np.zeros_like(orig_channel_sig_ts)
        unit_spike_train[channel_sig_temp.spike_times[unitOI].astype(int)] = 1

        # spike trains convolved with waveforms
        unit_spike_ts = np.convolve(unit_spike_train, unit_filters['average_waveform']['time_filter'][:truncate_idx])[20:-(filter_add_length-1)]
        unit_spike_ts_list.append(unit_spike_ts)

        # add filter PSDs to list
        filter_psds.append(unit_filters['average_waveform']['filter_psd'])

        # add firing rates to list
        sel_firing_rates.append(channel_sig_temp.firing_rates[unitOI])

spike_ts_sum = np.sum(unit_spike_ts_list, axis=0)
spike_ind_sum = np.sum(unit_spike_indicators, axis=0)
filter_psds = np.array(filter_psds)
sel_firing_rates = np.array(sel_firing_rates)
n_units = sel_firing_rates.size

freq_range = (300,5000)

if ts_mode == 'spikes':
    ts_signal = spike_ts_sum
elif ts_mode == 'raw':
    ts_signal = orig_channel_sig_ts

# calculate S_channel
tw = 3.5 # 3.5 works well for 1 thingy
n_taps = 2*tw - 1

mt_PSD,mt_freqs = fn.spectrafuncs.multitaper_psd(ts_signal, fs=fs, thbp=tw, start_time=time_window[0])

theor_spk_PSD_units = filter_psds * sel_firing_rates[:,None]
theor_spk_PSD = np.sum(theor_spk_PSD_units, axis=0)

# theor_spk_PSD = filter_psds[0] * sel_firing_rates[0]

# QQ plot
mt_PSD_trunc,mt_freqs_trunc = fn.spectrafuncs.spectrum_trunc(mt_freqs,mt_PSD,freq_range)
theor_spk_PSD_trunc,theor_freqs_trunc = fn.spectrafuncs.spectrum_trunc(theor_freqs,theor_spk_PSD,freq_range)
theor_spk_PSD_interp = fn.spectrafuncs.spectrum_interp(mt_freqs_trunc, theor_freqs_trunc, theor_spk_PSD_trunc)

Z_f = np.sort((n_taps * mt_PSD_trunc) / theor_spk_PSD_interp)
n = Z_f.size

theor_quant = stats.gamma.ppf((np.arange(1, n + 1) - 0.5) / n, a=n_taps)

freq_range2 = (300,1000)

mt_PSD_trunc2,mt_freqs_trunc2 = fn.spectrafuncs.spectrum_trunc(mt_freqs,mt_PSD,freq_range2)
theor_spk_PSD_trunc2,theor_freqs_trunc2 = fn.spectrafuncs.spectrum_trunc(theor_freqs,theor_spk_PSD,freq_range2)
theor_spk_PSD_interp2 = fn.spectrafuncs.spectrum_interp(mt_freqs_trunc2, theor_freqs_trunc2, theor_spk_PSD_trunc2)

Z_f2 = np.sort((n_taps * mt_PSD_trunc2) / theor_spk_PSD_interp2)
n2 = Z_f2.size

theor_quant2 = stats.gamma.ppf((np.arange(1, n2 + 1) - 0.5) / n2, a=n_taps)

gridspec = {'width_ratios': [1, 1, 0.4, 0.4]}
fig, ax = plt.subplots(1,4, figsize=(10,2), gridspec_kw=gridspec)

ax = ax.flatten()
ts_sig_nan = ts_signal.copy()
ts_sig_nan[spike_ind_sum != 0] = np.nan
ax[0].plot(orig_channel_sig_ta, ts_sig_nan, color='k', linewidth=0.7)
for i in range(n_units):
    unit_spk_sig_nan = ts_signal.copy()
    unit_spk_sig_nan[unit_spike_indicators[i] == 0] = np.nan
    ax[0].plot(orig_channel_sig_ta,unit_spk_sig_nan,
                color=MISC6_PAL[i], linewidth=0.7)
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Voltage (mV)")
ax[0].set_xlim((203.5, 209.5))
ax[0].set_xticks([204,206,208,209])
ax[0].set_xticklabels([204,206,208,''])
ax[0].set_yticks([-125, 0, 125])
ax[0].set_ylim((-150, 150))
ax[0].spines['bottom'].set_bounds(204, 209)
ax[0].spines['left'].set_bounds(-125, 125)

ax[1].semilogy(mt_freqs, mt_PSD, color='k', linewidth=0.6)
for i,theor_spk_PSD_unit in enumerate(theor_spk_PSD_units):
    ax[1].semilogy(theor_freqs, theor_spk_PSD_unit,linewidth=0.8,
                color=MISC6_PAL[i])
    
if n_units > 1:
    theor_psd_plot_label = r'Theoretical PSD $\left( \sum_\mathrm{units} \lambda^{\mathrm{unit}}|\mathcal{H}^\mathrm{AP,unit}|^2\right)$'
    theor_color = 'darkgray'
else:
    theor_psd_plot_label = r'Theoretical PSD $\left( \lambda^{\mathrm{unit}}|\mathcal{H}^\mathrm{AP,unit}|^2\right)$'
    theor_color = MISC6_PAL[0]
ax[1].semilogy(theor_freqs, theor_spk_PSD,
             color=theor_color, label=theor_psd_plot_label)
ax[1].legend(fontsize='x-small', loc='lower left')
ax[1].set_xlim(250, 5050)
ax[1].set_ylim(6e-9, 1)
ax[1].set_xticks([300, 1000, 2500, 5000])
ax[1].set_yticks([1e-8,1e-6,1e-4,1e-2,1])
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Log Power")
ax[1].spines['left'].set_bounds(1e-8, 1)
ax[1].spines['bottom'].set_bounds(300, 5e3)

ax[2].scatter(theor_quant, Z_f, marker='x', c='k', s=1)
ax[2].plot(np.arange(-100,100), np.arange(-100,100), color='red', linewidth=0.5)
ax[2].set_xlim((-1, 21))
ax[2].set_ylim((-1, 21))
ax[2].set_xlabel("Theoretical quantiles")
ax[2].set_ylabel("Sample quantiles")
ax[2].set_xticks([0,10,20])
ax[2].spines['bottom'].set_bounds(0, 20)
ax[2].spines['left'].set_bounds(0, 20)

ax[3].scatter(theor_quant2, Z_f2, marker='x', c='k', s=1)
ax[3].plot(np.arange(-100,100), np.arange(-100,100), color='red', linewidth=0.5)
ax[3].set_xlim((-1, 21))
ax[3].set_ylim((-1, 21))
ax[3].set_xlabel("Theoretical quantiles\n (300-1000 Hz)")
ax[3].set_ylabel("Sample quantiles\n (300-1000 Hz)")
ax[3].set_xticks([0,10,20])
ax[3].spines['bottom'].set_bounds(0, 20)
ax[3].spines['left'].set_bounds(0, 20)

fig.tight_layout()
fig.savefig(f"figs/tri_sig_plot_ch{channel_OI}{ts_mode}.png", dpi=600)

# with matplotlib.backends.backend_pdf.PdfPages(f'figs/tri_sig_plot_ch{channel_OI}{ts_mode}.pdf') as pdf:
#     fig.set_dpi(600)
#     fig.savefig(pdf, format='pdf', dpi=600, rasterized=True)

plt.show(block=True)