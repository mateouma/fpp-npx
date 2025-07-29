import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

import fppnpx as fn
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
# channel_signals = gen_all_channel_signals(signal_dataset, 300, False)


wavemap_cluster_labels2 = np.load("chkdelay_dlpfc_0630_wavemap_clusters_0521_res_1.npy")

unit_total_frs = np.array([signal_dataset['cluster_info'][signal_dataset['cluster_info']['cluster_id'] == u_id]['fr'] for u_id in signal_dataset['units']])
unit_total_frs = np.squeeze(unit_total_frs)

unit_depths = np.array([signal_dataset['cluster_info'][signal_dataset['cluster_info']['cluster_id'] == u_id]['depth'] for u_id in signal_dataset['units']])
unit_depths = np.squeeze(unit_depths)

wavemap_order_dict = {
    0: 0,
    1: 3,
    2: 5,
    3: 6,
    4: 2,
    5: 1,
    6: 4
}

fig, ax = plt.subplots(figsize=(7,5), dpi=300)

rand_widths = np.random.uniform(-0.3,0.3, size=138)
for i in range(138):
    ax.scatter(wavemap_order_dict[wavemap_cluster_labels2[i]]+1+rand_widths[i], unit_depths[i],
                #edgecolor=WAVEMAP_PAL[wavemap_cluster_labels2[i]],
                #marker=wfpath_list[i],
                marker='o', facecolor=WAVEMAP_PAL[wavemap_cluster_labels2[i]],edgecolors='white', linewidths=1,
                #facecolor='none',
                s=7.5*unit_total_frs[i])
    
ax.scatter(-100, -10, marker='o', facecolor='k', edgecolor='white', linewidths=1, s=(7.5), label='1 Hz')
ax.scatter(-110, -15, marker='o', facecolor='k', edgecolor='white', linewidths=1, s=(15*7.5), label='15 Hz')
ax.scatter(-111, -11, marker='o', facecolor='k', edgecolor='white', linewidths=1, s=(30*7.5), label='30 Hz')

ax.set_ylim((-45.0, 4025.0))
ax.set_xlim((0.3805, 7.54))
ax.set_xlabel("WaveMAP cluster")
ax.set_ylabel("Distance along shank (µm)")
ax.spines['bottom'].set_bounds(1, 7)
ax.spines['left'].set_bounds(0, 4000)
ax.set_xticks([1,2,3,4,5,6,7])
ax.set_xticklabels(['BS-1', 'BS-2', 'NS-1', 'NS-2', 'TP-1', 'TP-2', 'PS-1'])
ax.legend()
#ax.grid(True)

fig.savefig("figs/laminar_cluster_frs.svg", dpi=300)

plt.show(block=True)