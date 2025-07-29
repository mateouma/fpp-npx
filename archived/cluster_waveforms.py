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

mean_unit_waveforms = np.array(signal_dataset['waveform_info']['waveforms']).mean(axis=2)

wmc = WaveMAPClassifier(mean_unit_waveforms)

f_wfs,arr_wfs = wmc.plot_waveforms()

wmc.compute_waveform_umap(rand_state=99)
wmc.apply_louvain_method(resolution=1.0)

f_umap,arr_umap = wmc.plot_umap(show_clustering_solution=True)

wmc.plot_groups()

f_wfs.savefig("figs/normalized_waveforms.svg", dpi=300)
f_umap.savefig("figs/wavemap_umap_res1.svg", dpi=300)

wavemap_cluster_labels = np.array(wmc.clustering_solution)
np.save('chkdelay_dlpfc_0630_wavemap_clusters_0521_res_1.npy', wavemap_cluster_labels)
