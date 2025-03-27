"""
Code adapted from Lee et al. 2021 "Non-linear dimensionality reduction on extracellular waveforms reveals
cell type diversity in premotor cortex"
"""

import os
import random
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
import scipy
from scipy import io
import pickle as pkl
import h5py
import xml.etree.ElementTree as ET
import networkx as nx
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from umap import umap_ as umap
from community import community_louvain as louvain
import shap

class WaveMAPClassifier:

    def __init__(self, waveforms, add_noise=False, noise_sd=0.1, cluster_palette=None):

        if cluster_palette == None:
            self.CLUSTER_PALETTE = ['#5e60ce', '#00c49a','#ffca3a','#D81159','#fe7f2d','#7bdff2','#0496ff','#efa6c9','#ced4da', '#1eb43a']
        else:
            self.CLUSTER_PALETTE = cluster_palette

        # standardize by extrema
        waveforms = (waveforms - waveforms.mean(axis=1)[:,None]) / np.abs(waveforms).max(axis=1)[:,None]

        if add_noise:
            noise = np.zeros(waveforms.shape)
            for i,row in enumerate(noise):
                noise[i,:] = np.random.normal(0, noise_sd, waveforms.shape[1])
            waveforms += noise
        self.waveforms = waveforms

        self.UMAP_FIT = False
        self.LOUVAIN_APP = False

    def compute_waveform_umap(self, n_neighbors=15, min_dist=0.1, rand_state=42):
        print(f"Computing UMAP: {n_neighbors} neighbors | {min_dist} minimum distance | random state {rand_state}")

        # set seed
        np.random.seed(rand_state)
        os.environ['PYTHONHASHSEED'] = str(rand_state)
        random.seed(rand_state)        

        # compute UMAP
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=rand_state)
        mapper = reducer.fit(self.waveforms)
        embedding = reducer.transform(self.waveforms)
        umap_df = pd.DataFrame(embedding, columns=('x', 'y'))
        umap_df['waveform'] = list(self.waveforms)

        self.mapper = mapper
        self.umap_df = umap_df

        self.UMAP_FIT = True
        
        print("UMAP computed!")
        
    def apply_louvain_method(self, resolution=1.5):
        print(f"Applying Louvain method: {resolution} resolution")
        G = nx.from_scipy_sparse_array(self.mapper.graph_)
        clustering = louvain.best_partition(G, resolution=resolution)
        clustering_solution = list(clustering.values())
        self.n_clust = len(set(clustering_solution))

        self.LOUVAIN_APP = True

        print(f"{self.n_clust} clusters detected.")

        self.umap_df['color'] = clustering_solution
        self.clustering_solution = clustering_solution
        self.cluster_colors = [self.CLUSTER_PALETTE[i] for i in clustering_solution]

        cluster_waveforms = []
        for label_ix in range(self.n_clust):
            group_ixs = [i for i,x in enumerate(clustering_solution) if x == label_ix]
            group_waveforms = self.umap_df.iloc[group_ixs]['waveform'].tolist()
            cluster_waveforms.append(group_waveforms)            
        self.cluster_waveforms = cluster_waveforms
        

    def plot_umap(self, show_clustering_solution=False):
        f,arr = plt.subplots(1, figsize=[7,4.5], tight_layout={'pad': 0})
        f.tight_layout()

        if not show_clustering_solution :
            arr.scatter(self.umap_df.x, self.umap_df.y, s=10, c='k')
        else:
            try:
                arr.scatter(self.umap_df['x'].tolist(), self.umap_df['y'].tolist(), 
                        marker='o', c=self.cluster_colors, s=10, edgecolor='w',
                        linewidth=0.5)
            except AttributeError:
                print("Louvain clusters must be computed first.")
        
        arr.spines['top'].set_visible(False)
        arr.spines['bottom'].set_visible(False)
        arr.spines['left'].set_visible(False)
        arr.spines['right'].set_visible(False)
        arr.set_xticks([]);
        arr.set_yticks([]);

        # make this editable later
        arr.arrow(-3,0.8,0,1.5, width=0.05, shape="full", ec="none", fc="black")
        arr.arrow(-3,0.8,1.2,0, width=0.05, shape="full", ec="none", fc="black")

        arr.text(-3,0.3,"UMAP 1", va="center")
        arr.text(-3.5,1.0,"UMAP 2",rotation=90, ha="left", va="bottom")

    def plot_waveforms(self):
        f,arr = plt.subplots(1)
        arr.plot(self.waveforms.T,c='k',alpha=0.05);
        f.tight_layout()
        arr.spines['left'].set_visible(False)
        arr.spines['right'].set_visible(False)
        arr.spines['top'].set_visible(False)
        arr.spines['bottom'].set_visible(False)

    def plot_groups(self, mean_only=False, detailed=False, group='all'):
        """
        TO-DO: move this outside the classifier
        """
        if group == 'all':
            group = range(self.n_clust)
        else:
            group = [group]

        for label_ix in group:
            #plot_group(i,clustering_solution,umap_df,CUSTOM_PAL_SORT_3)
            f,arr = plt.subplots()
            f.set_figheight(1.8*1)
            f.set_figwidth(3.0*1)

            if not mean_only:
                for i,group_waveform in enumerate(self.cluster_waveforms[label_ix]):
                    arr.plot(group_waveform, c=self.CLUSTER_PALETTE[label_ix], alpha=0.3, linewidth=1.5)
                arr.plot(np.mean(self.cluster_waveforms[label_ix], axis=0), c='k', linestyle='-')
            else:
                arr.plot(np.mean(self.cluster_waveforms[label_ix], axis=0), c=self.CLUSTER_PALETTE[label_ix])

            arr.spines['right'].set_visible(False)
            arr.spines['top'].set_visible(False)

            if detailed:
                avg_peak = np.mean([np.argmax(x) for x in self.cluster_waveforms[label_ix][14:]])
                arr.axvline(avg_peak,color='k',zorder=0)
                
                arr.set_ylim([-5,5])
                arr.set_yticks([])
                #arr.set_xticks([0,7,14,21,28,35,42,48])
                arr.tick_params(axis='both', which='major', labelsize=12)
                #arr.set_xticklabels([0,'',0.5,'',1.0,'',1.5,''])
                arr.spines['left'].set_visible(False)
                arr.grid(False)
                #arr.set_xlim([0,48])
            else:
                arr.set(xticks=[],yticks=[])

            if not mean_only:
                x,y = 2.1,0.6
                ellipse = mpl.patches.Ellipse((x,y), width=5.4, height=0.3, facecolor='w',
                                    edgecolor='k',linewidth=1.5)
                label = arr.annotate(str(label_ix+1), xy=(x-0.25, y-0.08),fontsize=12, color = 'k', ha="center")
                arr.add_patch(ellipse)

                if i != -1:
                    x, y = 40,-1
                    n_waveforms = plt.text(x, y, 
                                        'n = '+str(len(self.cluster_waveforms[label_ix]))+
                                        ' ('+str(round(len(self.cluster_waveforms[label_ix])/len(self.umap_df)*100,2))+'%)'
                                        , fontsize=10)

            f.show()

    def save_results(self):
        pass