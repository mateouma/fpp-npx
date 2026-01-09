import numpy as np
import matplotlib.pyplot as plt
import mat73 as mt
import scipy.io as sio
import matplotlib.pyplot as plt

from fppnpx.preprocessing import read_session, load_waveforms, read_bin
from fppnpx.ChannelSignal import ChannelSignal
from fppnpx.plottools import WAVEMAP_PAL2
from fppnpx.filters import gen_filter
from fppnpx.unit_clustering import WaveMAP

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

# create Neuropixels session dict
imecpath = "/Users/mateouma/Desktop/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0"
npxsession = read_session(time_window=[100,123], imec_path=imecpath)

# load waveforms from output .mat file
waveformpath = "/Users/mateouma/Downloads/monkey datasets/20230630_DLPFCwaveforms.mat"
waveform_dataset, waveform_means = load_waveforms(waveformpath, npxsession)

# cluster (not wavemap cluster but spike sorted unit = cluster)
cluster_ids = np.array(list(waveform_dataset.keys()))

# perform WaveMAP on waveform means
wmc = WaveMAP(random_state=99, resolution=1.0, cluster_palette=WAVEMAP_PAL2)
wmc.fit(waveforms=waveform_means, verbose=True)

# plot and save UMAP with clustering solution
fig,ax = plt.subplots()
wmc.plot_umap(show_clustering_solution=True, ax=ax)
fig.savefig("figs/wavemap_umap.svg")

# plot and save waveforms
waveform_group_fig = wmc.plot_groups()
waveform_group_fig.savefig("figs/wavemap_waveforms.svg")

# plot and save cluster power spectra
fig,ax = plt.subplots()

# for each cluster: find each (mean) waveform of each unit within the cluster, compute the spectrum using FFT, and then average
cluster_idx_unit_list = []
for clust in np.unique(wmc.clustering_solution):
    wmc_clust_sol_idx = np.where(np.array(wmc.clustering_solution) == clust)[0]
    cluster_idx_unit_list.append(cluster_ids[wmc_clust_sol_idx])
    wmc_cluster_filters = []
    for wm_idx in wmc_clust_sol_idx:
        wf_filter_t,wf_filter_f,wf_filter_psd,freq_axis = gen_filter(wmc.waveforms[wm_idx], n=30000, fs=30000, center=True)
        wmc_cluster_filters.append(wf_filter_psd)
    cluster_filter = np.array(wmc_cluster_filters).mean(axis=0)
    ax.semilogy(freq_axis, cluster_filter, color=wmc.CLUSTER_PALETTE[clust], label=f"{clust+1}")
ax.legend()
ax.set_xlim(250, 5050)
ax.set_xticks([300, 1000, 2500, 5000])
ax.set_ylim((2e-11, 3e-7))
ax.spines['left'].set_bounds(3e-11, 3e-7)
ax.spines['bottom'].set_bounds(300, 5e3)
fig.savefig("figs/wavemap_power_spectra.svg")

# plot and save relative depth of each cluster
clust_order = [0, 3, 5, 6, 2, 1, 4]
fig,ax = plt.subplots()
for clust in np.unique(wmc.clustering_solution):
    cluster_unit_ids = cluster_idx_unit_list[clust]
    wmc_cluster_info = npxsession['cluster_info'][np.isin(npxsession['cluster_info']['cluster_id'],cluster_unit_ids)]
    ax.scatter(np.random.randn(len(cluster_unit_ids))/8 + clust_order[clust], wmc_cluster_info['depth'], color=wmc.CLUSTER_PALETTE[clust], s=wmc_cluster_info['fr']*7.5, edgecolor='white')
ax.set_ylim((-45.0, 4025.0))
ax.set_xlim((-1, 7))
ax.set_xlabel("WaveMAP cluster")
ax.set_ylabel("Distance along shank (µm)")
ax.spines['bottom'].set_bounds(0, 6)
ax.spines['left'].set_bounds(0, 4000)
ax.set_xticks([0,1,2,3,4,5,6])
ax.set_xticklabels(['BS-1', 'BS-2', 'NS-1', 'NS-2', 'TP-1', 'TP-2', 'PS-1'])
fig.savefig("figs/wavemap_cluster_depths.svg")