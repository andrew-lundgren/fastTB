import numpy as np
from numpy.fft import rfft, irfft, fft, ifft
import scipy.signal as sig

from pycbc.waveform import get_fd_waveform
from pycbc.filter import sigma, match, matched_filter

class WFEnv():
    def __init__(self, tlen, sample_rate, psd_function, f_lower):
        if not isinstance(tlen, int) or not isinstance(sample_rate, int):
            raise ValueError('tlen and sample_rate must be integers.')
        self.tlen = tlen
        self.srate = sample_rate
        self.Nsamp = tlen*sample_rate
        self.fNsamp = 1+tlen*sample_rate//2

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
    def template(self, mtot, q, chi, approximant='IMRPhenomXAS'):
        m1, m2 = mtot * q / (1+q), mtot / (1+q)
        hp_fd, _ = get_fd_waveform(approximant=approximant,
                                    mass1=m1, mass2=m2,
                                    spin1z=chi, spin2z=chi,
                                    delta_f=1/self.tlen, f_lower=self.low_freq_cutoff)
        hp_fd.resize(self.fNsamp)
        return hp_fd
    def hp_template(self, mtot, q, chi, hp_fac, approximant='IMRPhenomXAS'):
        """hp_fac is the highpass factor. 1 will highpass at 7200/mtotal Hz"""
        tmplt = self.template(mtot, q, chi, approximant=approximant)
        filt = sig.firwin(self.srate//4-1, [hp_fac*7200./mtot], pass_zero=False,
                            fs=self.srate, window='hann')
        filt.resize(self.Nsamp)
        ffilt = np.abs(rfft(filt))
        return ffilt*tmplt
