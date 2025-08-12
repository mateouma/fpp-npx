import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import pymc as pm
import pytensor.tensor as pt
import pytensor
pytensor.config.optdb__max_use_ratio = 24.0
pytensor.config.gcc__cxxflags = '-fbracket-depth=1024'

from .spectrafuncs import *

class FPPSSM:

    def __init__(self, time_series, time_axis, fs, thbp=5.75, win_dur=0.5, win_step=0.5):
        """
        Initialize filtered point process (homogeneous Poisson) state-space model
        """
        self.time_series = time_series
        self.time_axis = time_axis

        self.spectrogram = multitaper_spectrogram(time_series,fs, time_halfbandwidth_product=thbp,
                                                  window_duration=win_dur, window_step=win_step,
                                                  start_time=time_axis[0])

        self.is_model_fit = False

    def build_model(self, filter_psd, filter_frequencies, x_bar, freq_range):
        """
        State-space model with AR(1) deviations around a known baseline x_bar (TO-DO: establishj prior)
        """
        spectrogram_power_trunc,spectrogram_freqs_trunc = spectrum_trunc(self.spectrogram.frequencies,
                                                                         self.spectrogram.power().squeeze(),
                                                                         freq_range, True)
        
        self.spectrogram_trunc_power = spectrogram_power_trunc.squeeze()
        self.spectrogram_trunc_freqs = spectrogram_freqs_trunc
        n_time,n_freq = self.spectrogram_trunc_power.shape
        self.n_time = n_time

        self.theoretical_frequencies = filter_frequencies

        self.S_f = filter_psd
        filter_spectrum_trunc,filter_freqs_truncated = spectrum_trunc(filter_frequencies, filter_psd, freq_range)
        filter_spectrum_interp = spectrum_interp(spectrogram_freqs_trunc,
                                                 filter_freqs_truncated,
                                                 filter_spectrum_trunc)

        with pm.Model() as model:
            # AR(1) coefficient
            alpha = pm.Uniform("alpha", lower=0.3, upper=0.99)

            # sigma (noise standard deviation)
            sigma = pm.Exponential("sigma", lam=0.01)

            # initial deviation
            d_0 = pm.Normal("d_0", mu=0.0, sigma=100.0)

            # baseline firing rate
            #x_bar = pm.Uniform("x_bar", lower=500, upper=750)

            # build AR(1) dynamics
            d_states = [d_0]
            for t in range(1, n_time):
                d_t = pm.Normal(f"d_{t}", mu=alpha*d_states[t-1], sigma=sigma)
                d_states.append(d_t)
            d_stack = pt.stack(d_states)

            # reconstruct latent state (population firing rate)
            x_stack = d_stack + x_bar

            # compute mean for each time-frequency bin
            S_tf = x_stack[:, None] * filter_spectrum_interp[None,:]

            # gamma likelihood with shape=1
            y_obs = pm.Gamma("y_obs", alpha=1, beta=1/S_tf, observed=self.spectrogram_trunc_power)
        
        self.model = model

    def build_mv_model(self, filter_psds, filter_frequencies, x_bar, freq_range):
        """
        State-space model with AR(1) deviations around a known baseline x_bar (TO-DO: establishj prior)
        """
        spectrogram_power_trunc,spectrogram_freqs_trunc = spectrum_trunc(self.spectrogram.frequencies,
                                                                   self.spectrogram.power().squeeze(),
                                                                   freq_range, True)
        
        self.spectrogram_trunc_power = spectrogram_power_trunc.squeeze()
        self.spectrogram_trunc_freqs = spectrogram_freqs_trunc
        n_time,n_freq = self.spectrogram_trunc_power.shape
        self.n_time = n_time

        self.theoretical_frequencies = filter_frequencies

        self.S_f_list = filter_psds
        filter_spectrum_interp_list = []

        for i,filter_psd in enumerate(filter_psds):
            filter_spectrum_trunc,filter_freqs_truncated = spectrum_trunc(filter_frequencies, filter_psd, freq_range)
            filter_spectrum_interp = spectrum_interp(spectrogram_freqs_trunc,
                                                    filter_freqs_truncated,
                                                    filter_spectrum_trunc)
            filter_spectrum_interp_list.append(filter_spectrum_interp)
        filter_spectrum_interp_list = np.array(filter_spectrum_interp_list)
        self.S_f_interp_list = filter_spectrum_interp_list

        print("Instantiating model...")
        with pm.Model() as model:
            alpha_1 = pm.Uniform("alpha_1", lower=0.3, upper=0.99)
            alpha_2 = pm.Uniform("alpha_2", lower=0.3, upper=0.99)
            # alpha_3 = pm.Uniform("alpha_3", lower=0.3, upper=0.99)
            # alpha_4 = pm.Uniform("alpha_4", lower=0.3, upper=0.99)

            # sigma (noise standard deviation)
            sigma = pm.Exponential("sigma", lam=0.01)
            # sigma_1 = pm.Exponential("sigma_1", lam=0.01)
            # sigma_2 = pm.Exponential("sigma_2", lam=0.01)
            # sigma_3 = pm.Exponential("sigma_3", lam=0.01)
            # sigma_4 = pm.Exponential("sigma_4", lam=0.01)
            
            d_0 = pm.MvNormal("d_0", mu=np.ones(self.n_filts), cov=100.0*np.eye(self.n_filts))

            # baseline firing rate
            #x_bar = pm.Uniform("x_bar", lower=500, upper=750)

            # build AR(1) dynamics
            d_states = [d_0]
            alphas = pt.stack([alpha_1, alpha_2])#, alpha_3, alpha_4])

            for t in range(1, n_time):
                d_t = pm.MvNormal(f"d_{t}", mu=alphas*d_states[t-1], cov=sigma*np.eye(self.n_filts))
                d_states.append(d_t)
            d_stack = pt.stack(d_states)

            # reconstruct latent state (population firing rate)
            x_stack = d_stack + x_bar * pytensor.shared(np.ones(self.n_filts))

            S_tf = pt.dot(x_stack, pytensor.shared(filter_spectrum_interp_list))

            # gamma likelihood with shape=1
            y_obs = pm.Gamma("y_obs", alpha=1, beta=1/S_tf, observed=self.spectrogram_trunc_power)
        
        self.model = model
    print("Model instantiated")

    def fit(self, filter_psd, filter_frequencies, x_bar, freq_range=(1,6000)):
        self.build_model(filter_psd, filter_frequencies, x_bar, freq_range)
        model_b = self.model

        with model_b:
            trace_b = pm.sample(
                draws=1000, tune=1000, target_accept=0.9, random_seed=99
            )

        self.d_vars = ["sigma"] + [f"d_{t}" for t in range(self.n_time)]
        
        self.ssm_preds = [trace_b.posterior.data_vars[f'd_{i}'].mean().values for i in range(self.n_time)]
        
    def fit_mv(self, filter_psds, filter_frequencies, x_bar, freq_range=(1,6000)):
        self.n_filts = len(filter_psds)
        self.build_mv_model(filter_psds, filter_frequencies, x_bar, freq_range)
        # model_b = self.model

        with self.model:
            trace_b = pm.sample(
                draws=1000, tune=1000, target_accept=0.9, random_seed=99
            )

        self.d_vars = ["sigma"] + [f"d_{t}" for t in range(self.n_time)]
        self.trace_back = trace_b        
        self.ssm_preds = [trace_b.posterior.data_vars[f'd_{i}'].mean().values for i in range(self.n_time)]
        


