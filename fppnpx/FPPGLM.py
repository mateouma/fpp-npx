import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from .spectrafuncs import *

class FPPGLM:

    def __init__(self, time_series, time_axis, fs, n_tapers):
        """
        Initialize filtered point process (homogeneous poisson) generalized linear model

        TO-DO: make this apply to other point processes
        """
        self.filtered_point_process = time_series # observed signal
        self.time_axis = time_axis

        # calculate tapered power spectrum of the signal
        self.multitaper_psd,self.multitaper_frequencies = multitaper_psd(time_series, fs, n_tapers)

        self.is_model_fit = False

    def fit(self, filter_psd_list, lambda_observed, theoretical_frequencies, bias=False, mult_const=1, freq_range=(1,5000)):
        """
        Index and interpolate observed spectrum and filter spectra. Then fit Gamma GLM according
        to the formula y = XB where 'y' is the multitapered power spectrum, X is the design matrix
        containing the spectra of the filters, 
        """
        self.lambda_observed = lambda_observed
        self.theoretical_frequencies = theoretical_frequencies

        multitaper_psd_truncated,multitaper_frequencies_truncated = spectrum_trunc(self.multitaper_frequencies,
                                                                                  self.multitaper_psd,
                                                                                  freq_range)
        self.multitaper_psd_truncated = multitaper_psd_truncated
        self.multitaper_frequencies_truncated = multitaper_frequencies_truncated

        S_truncated = []
        S_interpolated = []

        self.S_list = filter_psd_list
        for i,filter_spectrum in enumerate(filter_psd_list):
            filter_spectrum_trunc,theoretical_frequencies_truncated = spectrum_trunc(theoretical_frequencies,
                                                                                    filter_spectrum, freq_range)
            filter_spectrum_interp = spectrum_interp(multitaper_frequencies_truncated,
                                                     theoretical_frequencies_truncated,
                                                     filter_spectrum_trunc)
            S_truncated.append(filter_spectrum_trunc)
            S_interpolated.append(filter_spectrum_interp)
        self.theoretical_frequencies_truncated = theoretical_frequencies_truncated
        self.S_truncated = np.array(S_truncated)
        self.S_interpolated = np.array(S_interpolated)

        # create design matrix
        X = self.S_interpolated.T
        self.bias = bias
        if bias:
            X = sm.add_constant(X)

        # response is multitapered signal spectrum
        self.mult_const = mult_const
        y = self.multitaper_psd_truncated * self.mult_const

        # fit statsmodels Gamma GLM
        glm_model = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.Identity()))
        glm_results = glm_model.fit()

        self.lambda_estimated =  glm_results.params
        
        phi = glm_results.scale
        k = 1/phi
        self.gamma_dist_params = {'phi':phi, 'k':k}       

        self.is_model_fit = True 

        if self.bias:
            reported_lambda = self.lambda_estimated[1]
        else:
            reported_lambda = self.lambda_estimated[0]

        print(f"Estimated Bias:               {self.lambda_estimated[0]:.3f}")
        print(f"Theoretical $\lambda_0$:      {lambda_observed[0]:.3f}")
        print(f"Estimated $\lambda_0$:        {reported_lambda:.3f}")
        print(f"Gamma Dispersion (phi):       {phi:.3f}")
        print(f"Gamma Shape (k):              {k:.3f}")

    def predict(self, interp_filters=None):
        if interp_filters is None:
            X = self.S_interpolated.T
        else:
            X = interp_filters.T
            
        if self.bias:
            X = sm.add_constant(X)
        
        self.glm_predicted_psd = X @ self.lambda_estimated
        return self.glm_predicted_psd

    def plot_results(self):
        if not self.is_model_fit:
            raise AttributeError("Must fit model before plotting results")

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=300)

        # (a) Filtered Time Series
        axes[0].plot(self.time_axis, self.filtered_point_process, color='k', lw=1)
        axes[0].set_title("(a) Time Series")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True)

        # (b) PSD Comparison
        # i. empirical PSD
        axes[1].loglog(
            self.multitaper_frequencies_truncated,
            self.multitaper_psd_truncated * self.mult_const,
            label="Multitaper PSD",
            linewidth=2,
            linestyle="--",
            color='darkgray'
        )

        # ii. theoretical PSD
        axes[1].loglog(
            self.theoretical_frequencies_truncated,
            np.array(self.S_truncated).T @ self.lambda_observed,
            label="Theoretical PSD (True $\lambda_0$)",
            linewidth=2,
            color='k'
        )

        # iii. predicted PSD using estimated rates
        axes[1].loglog(
            self.multitaper_frequencies_truncated,
            self.predict(),
            label="GLM PSD (Est. $\lambda_0$)",
            linewidth=3,
            color='red'
        )

        axes[1].set_title("(b) PSD Comparison")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Power")
        axes[1].grid(True)

        axes[1].legend(loc="best")
        plt.tight_layout()

        plt.show()

    def save_results(self):
        pass