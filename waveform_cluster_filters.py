import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

import fppnpx as fn
from fppnpx.signalfuncs import load_signal, gen_all_channel_signals
from wavemapnpx.WaveMAPClassifier import WaveMAPClassifier

from plottools import WAVEMAP_PAL2, ml_map, cmap

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

apath = "/Users/mateouma/Downloads/monkey datasets/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0_t0.exported.imec0.ap-001.bin"
cpath = "/Users/mateouma/Downloads/monkey datasets/cluster_info_task.tsv"
wpath = "/Users/mateouma/Downloads/monkey datasets/20230630_DLPFCwaveforms.mat"

cluster_labels = np.load('chkdelay_dlpfc_0630_wavemap_clusters_0521_res_1.npy')

fs = 30000
time_window = [204,209] # seconds within the recording # [100,123]

signal_dataset = load_signal(appath=apath, time_window=time_window, fs=fs, cipath=cpath, wfpath=wpath)
# channel_signals = gen_all_channel_signals(signal_dataset, 300, False)

mean_unit_waveforms = np.array(signal_dataset['waveform_info']['waveforms']).mean(axis=2)
mean_unit_waveforms_centered = (mean_unit_waveforms - mean_unit_waveforms.mean(axis=1)[:,None]) 
mean_unit_waveforms_scaled = mean_unit_waveforms_centered / np.abs(mean_unit_waveforms_centered).max(axis=1)[:,None]

def extract_mean_kernels(waveform_arr, cluster_labels):
    time_kernel_means = []
    frequency_kernel_means = []
    kernel_psd_means = []
    kernel_psd_ATKs = []

    for bn_lab_ix in range(np.unique(cluster_labels).size):
        group_waveforms = waveform_arr[cluster_labels == bn_lab_ix]

        time_kernels = []
        frequency_kernels = []
        kernel_psds = []

        for waveform in group_waveforms:
            kernel_t,kernel_f,kernel_psd,freq_axis = fn.filterfuncs.gen_filter(waveform, fs, fs, center=True)
            time_kernels.append(kernel_t)
            frequency_kernels.append(kernel_f)
            kernel_psds.append(kernel_psd)

        time_kernel_mean = np.array(time_kernels).mean(axis=0)
        freq_kernel_mean = np.array(frequency_kernels).mean(axis=0)
        kernel_psd_mean = np.array(kernel_psds).mean(axis=0)

        # PSD of the average time kernel across the group
        _,__,kernel_psd_ATK,___ = fn.filterfuncs.gen_filter(time_kernel_mean, fs, fs, center=True)

        time_kernel_means.append(time_kernel_mean)
        frequency_kernel_means.append(freq_kernel_mean)
        kernel_psd_means.append(kernel_psd_mean)
        kernel_psd_ATKs.append(kernel_psd_ATK)

    time_kernel_means = np.array(time_kernel_means)
    frequency_kernel_means = np.array(frequency_kernel_means)
    kernel_psd_means = np.array(kernel_psd_means)
    kernel_psd_ATKs = np.array(kernel_psd_ATKs)

    return time_kernel_means, frequency_kernel_means, kernel_psd_means, kernel_psd_ATKs, freq_axis

time_kernels_raw,freq_kernels_raw,kernel_psds_raw,kernel_psds_ATK_raw,freq_axis = extract_mean_kernels(mean_unit_waveforms, cluster_labels)
time_kernels_sc,freq_kernels_sc,kernel_psds_sc,kernel_psds_ATK_sc,freq_axis = extract_mean_kernels(mean_unit_waveforms_scaled, cluster_labels)


# np.save("wavemap_mean_waveforms_raw.npy", time_kernels_raw[:,:62])
# np.save("wavemap_mean_waveforms_scaled.npy", time_kernels_sc[:,:62])

np.save("wavemap_freq_axis.npy", freq_axis)
np.save("wavemap_mean_psds_raw.npy", kernel_psds_raw)
np.save("wavemap_mean_psds_scaled.npy", kernel_psds_raw)

# fig, ax = plt.subplots(1,2, figsize=(8,3))

# for clab in np.unique(cluster_labels):
#     ax[0].plot(time_kernels_raw[clab], color=WAVEMAP_PAL[clab], label=f"Cluster {clab}")
#     ax[1].loglog(freq_axis, kernel_psds_raw[clab], color=WAVEMAP_PAL[clab], label=f"Cluster {clab}")
# ax[0].set_xlim((0,62))
# ax[1].set_xlim((1,6000))
# ax[0].set_title("Mean Cluster Waveforms")
# ax[1].set_title("Mean Cluster Waveform PSDs")
# ax[1].legend()

cluster_labels_ordered = [0, 5, 4, 1, 6, 2, 3] # by color
waveform_labels = ['BS-1', 'BS-2', 'NS-1', 'NS-2', 'TP-1', 'TP-2', 'PS-1']

wf_time = np.linspace(0,1000,30000)

# normalized
fig, ax = plt.subplots(1,2, figsize=(8,3))

for wl,clab in enumerate(cluster_labels_ordered):
    # ax[0].plot(wf_time[:62],time_kernels_sc[clab,:62], color=WAVEMAP_PAL2[clab], label=f"Cluster {clab}")
    ax[1].semilogy(freq_axis, kernel_psds_sc[wl], color=WAVEMAP_PAL2[clab], label=waveform_labels[wl])


# ax[0].set_xlabel("Time (ms)")
ax[1].set_xlabel("Frequency (Hz)")
# ax[0].set_ylabel("Normalized Amplitude")
ax[1].set_ylabel("Log Power")
# ax[0].set_title("Mean Cluster Waveforms")
# ax[1].set_title("Mean Cluster Waveform PSDs")
ax[1].legend(fontsize='x-small')

# ax[0].set_ylim((-1.1,1.1))
# ax[0].set_xlim((-0.1,2.1))
# ax[0].set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
# ax[0].spines['bottom'].set_bounds(0, 2)
# ax[0].spines['left'].set_bounds(-1.0, 1.0)

ax[1].set_xlim(250, 5050)
ax[1].set_xticks([300, 1000, 2500, 5000])
ax[1].set_ylim((2e-11, 3e-7))
ax[1].spines['left'].set_bounds(3e-11, 3e-7)
ax[1].spines['bottom'].set_bounds(300, 5e3)
fig.tight_layout()

fig.savefig("figs/wavemap_filters.svg", dpi=300)

plt.show(block=True)