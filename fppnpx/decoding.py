import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy import optimize

from .spectral import *

class FPPGLM:

    def __init__(self):
        self.MODEL_FIT = False

    def fit(self, time_series, time_axis, filter_spectrum_list, filter_spectrum_frequencies,
            observed_rates, frequency_range, multitaper_args, verbose=False):
        """
        
        """
        self.observed_rates = observed_rates
        self.observed_time_series = time_series
        self.time_axis = time_axis

        # calculated multitapered power spectrum
        mtap_spectrum,mtap_frequencies = multitaper_spectrum(time_series, verbose=verbose, **multitaper_args)
        self.mtap_spectrum_truncated,self.mtap_frequencies_truncated = spectrum_trunc(mtap_frequencies, mtap_spectrum, frequency_range)

        # truncate and interpolate filter spectra
        self.S_list = filter_spectrum_list
        S_truncated = []
        S_interpolated = []
        for i,filter_spectrum in enumerate(filter_spectrum_list):
            filter_spectrum_trunc,filter_frequencies_trunc = spectrum_trunc(filter_spectrum_frequencies, filter_spectrum, frequency_range)
            filter_spectrum_interp = spectrum_interp(self.mtap_frequencies_truncated, filter_frequencies_trunc,filter_spectrum_trunc)
            
            S_truncated.append(filter_spectrum_trunc)
            S_interpolated.append(filter_spectrum_interp)
        self.filter_frequencies_truncated = filter_frequencies_trunc
        self.S_truncated = np.array(S_truncated)
        self.S_interpolated = np.array(S_interpolated)

        # create design matrix
        X = self.S_interpolated.T
        
        # response is multitapered signal spectrum
        y = self.mtap_spectrum_truncated

        # fit statsmodels Gamma GLM
        glm_results = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.Identity())).fit()

        self.estimated_rates = glm_results.params
        self.glm_results = glm_results

        # generate theoretical power spectra
        self.theoretical_spectrum_truncated = np.array(self.S_truncated).T @ self.observed_rates
        self.theoretical_spectrum_interpolated = spectrum_interp(self.mtap_frequencies_truncated, self.filter_frequencies_truncated, self.theoretical_spectrum_truncated)

        self.MODEL_FIT = True

        print(f"Observed rates:          {self.observed_rates}")
        print(f"Estimated rates:         {self.estimated_rates}")
        print(f"Gamma dispersion (phi):  {glm_results.scale}")
        print(f"Gamma shape (k):         {1 / glm_results.scale}")

    def predict(self, interp_filters=None):
        if interp_filters is None:
            X = self.S_interpolated.T
        else:
            X = interp_filters.T

        self.predicted_spectrum = X @ self.estimated_rates

    
        