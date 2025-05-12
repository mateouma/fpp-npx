import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import pymc as pm
import pytensor.tensor as pt
import pytensor
pytensor.config.optdb__max_use_ratio = 24.0

from .spectrafuncs import *

class FFPSSM:

    def __init__(self, time_series, time_axis, fs, thbp=5.75, win_dur=0.5, win_step=0.5):
        """
        Initialize filtered point process (homogeneous Poisson) state-space model
        """
        self.time_series = time_series
        self.time_axis = time_axis

        self.spectrogram = multitaper_spectrogram(time_series,fs, time_halfbandwidth_product=thbp
                                                  window_duration=win_dur, window_step=win_step,
                                                  start_time=time_axis[0])

        self.is_model_fit = False

    # def fit(self, filter_psd, rate_observed, theoretical_frequencies, freq_range=(1,6000)):
    #     """
    #     State-space model with AR(1) deviations around a known baseline x_bar (TO-DO: establishj prior)
    #     """


    #     n_time,n_freq = y_data.shape

    #     self.rate_observed = rate_observed
    #     self.theoretical_frequencies = theoretical_frequencies

    #     self.S_f = filter_psd
    #     filter_spectrum_trunc,theoretical_frequencies_truncated = spectrum_trunc(theoretical_frequencies,
    #                                                                              filter_psd, freq_range)
    #     filter_spectrum_interp = spectrum_interp(multi)

    #     with pm.Model() as model:
    #         # AR(1) coefficient
    #         alpha = pm.Uniform("alpha", lower=0.3, upper=0.99)

    #         # noise
    #         sigma = pm.Exponential("sigma", lam=0.01)

    #         # initial deviation
    #         d_0 = pm.Normal("d_0", mu=0.0, sigma=100.0)

    #         # "baseline" firing rate
    #         x_i = pm.Uniform("x_i", lower=2, upper=30) # change this at some point?

    #         # AR(1) dynamics
    #         d_states = [d_0]
    #         for t in range(1, n_time):
    #             d_t = pm.Normal(f"d_{t}", mu=alpha*d_states[t-1], sigma=sigma)
    #             d_states.append(d_t)
    #         d_stack = pt.stack(d_states) # shape (n_time,)

    #         # reconstruct latent state
    #         x_stack = d_stack + x_i

    #         # compute mean for each time-frequency bin
    #         S_tf = x_stack[:,None] * filter_psd_interp[None,:]
            
    #         # Gamma likelihood with shape=1
    #         y_obs = pm.Gamma("y_obs", alpha=1, beta=1/S_tf, observed=y_data)

    #         trace_b = pm.sample(
    #             draws=1000, tune=1000, target_accept=0.9, random_seed=99
    #         )

    #     self.d_vars = ["sigma"] + [f"d_{t}" for t in range(n_time)]
        
    #     self.ssm_preds = [trace_b.posterior.data_vars[f'd_{i}'].mean().values for i in range(n_time)]
        


