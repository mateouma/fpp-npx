from scipy.io import savemat
from fppnpx.signalfuncs import load_signal, gen_all_channel_signals

appath = "/Users/mateouma/Downloads/monkey datasets/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0_t0.exported.imec0.ap-001.bin"
lfppath = "/Users/mateouma/Downloads/monkey datasets/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0/TIBERIUS_CHKDLAY_DLPFC_NPIX45_063023_g0_t0.exported.imec0.lf.bin"
cpath = "/Users/mateouma/Downloads/monkey datasets/cluster_info_task.tsv"
wpath = "/Users/mateouma/Downloads/monkey datasets/20230630_DLPFCwaveforms.mat"

fs_lfp = 2500
fs_ap = 30000
time_window = [204,209]

# do later

signal_dataset_AP = load_signal(appath=appath, time_window=time_window, fs=fs_ap, cipath=cpath, wfpath=wpath)
signal_dataset_LFP = load_signal(appath=lfppath, time_window=time_window, fs=fs_lfp, cipath=cpath, wfpath=wpath)

channel_signals_AP = gen_all_channel_signals(signal_dataset_AP)
channel_signals_LFP = gen_all_channel_signals(signal_dataset_LFP)

savemat(f'signal_info_{time_window[0]}-{time_window[1]}s_AP.mat', signal_dataset_AP)
savemat(f'signal_info_{time_window[0]}-{time_window[1]}s_LFP.mat', signal_dataset_LFP)
