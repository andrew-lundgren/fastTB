import numpy as np
from numpy.fft import rfft, irfft, fft, ifft
import scipy.signal as sig

from pycbc.filter import sigma, match, matched_filter

class WFEnv():
    def __init__(self, tlen, sample_rate, psd_function, f_lower):
        if not isinstance(tlen, int) or not isinstance(sample_rate, int):
            raise ValueError('tlen and sample_rate must be integers.')
        self.tlen = tlen
        self.srate = sample_rate
        self.Nsamp = tlen*sample_rate

        self.times = np.arange(self.Nsamp)/self.srate

        self.low_freq_cutoff = f_lower
        self.psd = psd_function(1+self.Nsamp//2, delta_f=1./tlen,
                                    low_freq_cutoff=f_lower)
        self.invasd = np.where(self.psd.sample_frequencies < self.low_freq_cutoff,
                               0., self.psd**-0.5)
        self.invasd[-1] = 0.
    def snr(self, waveform):
        return sigma(waveform,
                     psd=self.psd, low_frequency_cutoff=self.low_freq_cutoff)
    def match(self, waveform1, waveform2, return_time=False):
        my_match = match(waveform1, waveform2,
                            psd=self.psd, low_frequency_cutoff=self.low_freq_cutoff)
        if return_time:
            return my_match[0], my_match[1]/self.srate
        else:
            return my_match[0]
    def whiten(self, waveform):
        fwave = waveform.to_frequencyseries()
        fwave *= self.invasd
        return fwave.to_timeseries()
    def align(self, template, waveform):
        """Aligns a template to a given waveform"""
        snr = matched_filter(template, waveform,
                     psd=self.psd, low_frequency_cutoff=self.low_freq_cutoff)
        peak = abs(snr).numpy().argmax()
        snrp = snr[peak]
        time = snr.sample_times[peak]
        aligned = template.cyclic_time_shift(time - waveform.start_time)
        norm = snrp/self.snr(template)
        return (norm*aligned)
    def project(self, template, waveform):
        """Aligns template to waveform and tries to scale to same amplitude"""
        aligned = self.align(template, waveform)
        return aligned/self.match(template, waveform)