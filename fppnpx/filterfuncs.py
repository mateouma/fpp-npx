import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colormaps


def gen_filter(waveform, n, fs, truncate_idx=62, truncate_val=None, center=False):
    waveform = waveform[:truncate_idx]

    filter_t = np.zeros((n))
    filter_t[:truncate_idx] = waveform

    if truncate_val == None:
        truncate_val = waveform[-1]

    filter_t[truncate_idx:] = truncate_val

    if center:
        filter_t -= truncate_val

    wf_fft = np.fft.fft(filter_t) / n

    freq_idx = np.fft.fftfreq(n, 1/fs)
    sort_fftf = np.argsort(freq_idx)

    filter_f = wf_fft[sort_fftf]
    filter_psd = (np.abs(filter_f))**2

    freq_axis = np.linspace(-int(fs/2), int(fs/2), n)

    return filter_t, filter_f, filter_psd, freq_axis

def extract_mean_filters(cluster_waveforms, n_clust, n, fs):
    """
    To use in conjunction with WaveMAP
    """
    time_filter_means = []
    frequency_filter_means = []
    filter_psd_means = []
    filter_psd_ATKs = []

    for label_ix in range(n_clust):
        time_filters = []
        frequency_filters = []
        filter_psds = []

        for waveform in cluster_waveforms[label_ix]:
            filter_t,filter_f,filter_psd,freq_axis = gen_filter(waveform, n, fs, center=True)
            time_filters.append(filter_t)
            frequency_filters.append(filter_f)
            filter_psds.append(filter_psd)
        
        time_filter_mean = np.array(time_filters).mean(axis=0)
        freq_filter_mean = np.array(frequency_filters).mean(axis=0)
        filter_psd_mean = np.array(filter_psds).mean(axis=0)

        _, __,filter_psd_ATK,___ = gen_filter(time_filter_mean, n, fs, center=True)
        
        time_filter_means.append(time_filter_mean)
        frequency_filter_means.append(freq_filter_mean)
        filter_psd_means.append(filter_psd_mean)
        filter_psd_ATKs.append(filter_psd_ATK)

    mean_filters = {
        "time_filters":np.array(time_filter_means),
        "freq_filters": np.array(frequency_filter_means),
        "filter_psds": np.array(filter_psd_means),
        "filter_psds_ATK": np.array(filter_psd_ATKs)
    }

    return mean_filters, freq_axis

def plot_filters(filter_t, filter_f, filter_psd, freq_axis):
    fig, ax = plt.subplots(2, 2, dpi=300)
    # gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])  # adjust height ratio if needed

    # # Top row
    # ax1 = fig.add_subplot(gs[0, 0])  # Top-left
    # ax2 = fig.add_subplot(gs[0, 1])  # Top-right

    # # Bottom spans full width
    # ax3 = fig.add_subplot(gs[1, :])  # Bottom row (full width)
    ax1 = ax[0,0]
    ax2 = ax[0,1]
    ax3 = ax[1,0]

    # Your plots
    ax1.plot(filter_t, color='k')
    ax1.set_title('Time domain filter')
    ax1.set_xlim((0, 100))

    ax2.plot(freq_axis, filter_f.real, color='k', label='Real component')
    ax2.plot(freq_axis, filter_f.imag, color='k', linestyle='--', label='Imaginary component')
    ax2.set_title('Frequency domain filter')

    ax3.loglog(freq_axis, filter_psd, color='k')
    #ax3.set_xlim((0, 6000))
    ax3.set_title('Filter power spectral density')

    plt.tight_layout()


def plot_ia_filters(instances, filters, theor_freqs):
        """
        Plots the average waveform filters, average filter for each instance of the waveform, and each instance of the waveform
        """
        # create colormap
        cmap = colormaps.get_cmap('GnBu')
        cmap_idx = np.linspace(0,1.0, instances['spikes'])
        colors = cmap(cmap_idx)

        fig, ax = plt.subplots(1,2, figsize=(6,2), dpi=300)
        
        for i,color in enumerate(colors):
            ax[0].plot(instances['waveforms'][i], alpha=0.3, color=color)
        ax[0].plot(filters['average_waveform']['time_filter'], color='#e38710', label='Waveform-averaged')
        ax[0].plot(filters['waveform_instance']['time_filter'], color='#edc161', label='Instance-averaged')
        ax[0].plot()
        ax[0].legend()
        ax[0].set_xlim((0,100))

        for i,color in enumerate(colors):
            ax[1].loglog(theor_freqs, instances['psds'][i], alpha=0.3, color=color)
        ax[1].loglog(theor_freqs, filters['average_waveform']['filter_psd'], color='#e38710', label='Waveform-averaged')
        ax[1].loglog(theor_freqs, filters['waveform_instance']['filter_psd'], color='#edc161', label='Spectrum instance-averaged')
        ax[1].loglog(theor_freqs, filters['waveform_instance']['filter_psd_iaw'], color='coral', label='Waveform-average spectrum')
        ax[1].legend()
        #plt.xlim((1,5000))
        #plt.ylim((1e-10, 1e-2))
        #plt.show()

