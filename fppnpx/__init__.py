
# def gen_kernel(waveform, n, fs, truncate_idx=62, truncate_val=None, center=False):
#     waveform = waveform[:truncate_idx]

#     kernel_t = np.zeros((n))
#     kernel_t[:truncate_idx] = waveform

#     if truncate_val == None:
#         truncate_val = waveform[-1]

#     kernel_t[truncate_idx:] = truncate_val

#     if center:
#         kernel_t -= truncate_val

#     wf_fft = np.fft.fft(kernel_t) / n

#     freq_idx = np.fft.fftfreq(n, 1/fs)
#     sort_fftf = np.argsort(freq_idx)

#     kernel_f = wf_fft[sort_fftf]
#     kernel_psd = (np.abs(kernel_f))**2

#     freq_axis = np.linspace(-int(fs/2), int(fs/2), n)

#     return kernel_t, kernel_f, kernel_psd, freq_axis

# def plot_kernels(kernel_t, kernel_f, kernel_psd, freq_axis):
#     fig, axs = plt.subplots(2, 2, dpi=300)

#     ax1 = axs[0, 0]  # Top-left
#     ax2 = axs[0, 1]  # Top-right

#     fig.delaxes(axs[1, 1])

#     ax3 = fig.add_subplot(2, 1, 2)

#     ax1.plot(kernel_t, color='k')
#     ax1.set_title('Time domain kernel')
#     ax1.set_xlim((0,100))

#     ax2.plot(freq_axis, kernel_f, color='k')
#     ax2.set_title('Frequency domain kernel')

#     ax3.plot(freq_axis, kernel_psd, color='k')
#     ax3.set_title('Kernel power spectral density')

#     plt.tight_layout()
#     plt.show()

# def plot_ia_kernels(waveform_instances, kernel_psd_instances, aw_kernels, wi_kernels,
#                     theor_freqs, spk_instances, kernel_type=None):
#         """
#         Plots the average waveform kernels, average kernel for each instance of the waveform, and each instance of the waveform
#         """
#         # create colormap
#         cmap = colormaps.get_cmap('GnBu')
#         cmap_idx = np.linspace(0,1.0,spk_instances)
#         colors = cmap(cmap_idx)
        
#         if kernel_type == 'time':
#             fig, ax = plt.subplots(dpi=300)
#             for i,color in enumerate(colors):
#                 ax.plot(waveform_instances[i], alpha=0.3, color=color)
#             plt.plot(aw_kernels[f'time_kernel'], color='#e38710', label='Waveform-averaged')
#             plt.plot(wi_kernels[f'time_kernel'], color='#edc161', label='Instance-averaged')
#             plt.plot()
#             plt.legend()
#             plt.xlim((0,100))
#             plt.show()

#         elif kernel_type == 'psd':
#             fig, ax = plt.subplots(dpi=300)
#             for i,color in enumerate(colors):
#                 ax.loglog(theor_freqs, kernel_psd_instances[i], alpha=0.3, color=color)
#             plt.loglog(theor_freqs, aw_kernels['kernel_psd'], color='#e38710', label='Waveform-averaged')
#             plt.loglog(theor_freqs, wi_kernels['kernel_psd'], color='#edc161', label='Instance-averaged')
#             plt.loglog(theor_freqs, wi_kernels['kernel_psd_iaw'], color='coral', label='I-A, avg. waveform')
#             plt.legend()
#             #plt.xlim((1,5000))
#             #plt.ylim((1e-10, 1e-2))
#             plt.show()

# def plot_spectrum(x, fs, time_halfbandwidth_product=4, n_tapers=7, start_time=0.0, axes='loglog'):
#     multitaper = Multitaper(
#         x,
#         sampling_frequency=fs,
#         time_halfbandwidth_product=time_halfbandwidth_product,
#         n_tapers=n_tapers,
#         start_time=start_time
#     )
#     connectivity = Connectivity.from_multitaper(multitaper)

#     multitaper_psd = connectivity.power().squeeze()
#     multitaper_frequencies = connectivity.frequencies

#     plt.figure(dpi=300)
#     if axes=='loglog':
#         plt.loglog(multitaper_frequencies, multitaper_psd, linewidth=0.6, color='k')
#     else:
#         plt.plot(multitaper_frequencies, multitaper_psd, linewidth=0.6, color='k')
#     #plt.xlim((0,2000))
#     plt.show()

# def plot_spectrogram(x, fs, spk_train=None, time_halfbandwidth_product=4, window_duration=0.05, window_step=0.025, start_time=0.0):
#     fig, ax = plt.subplots()
#     multitaper = Multitaper(
#         x,
#         sampling_frequency=fs,
#         time_halfbandwidth_product=time_halfbandwidth_product,
#         time_window_duration=window_duration,
#         time_window_step=window_step,
#         start_time=start_time,
#     )
#     connectivity = Connectivity.from_multitaper(multitaper)
#     im = ax.pcolormesh(
#         connectivity.time,
#         connectivity.frequencies,
#         connectivity.power().squeeze().T,
#         cmap="viridis",
#         shading="auto",
#     )
#     plt.ylim((0, 2000))

#     if spk_train is not None:
#         plt.scatter(spk_train / fs, np.repeat(250, len(spk_train)), color='red', marker='^', s=3)
#     ax.set_ylabel("Frequency")
#     ax.set_xlabel("Time")