import numpy as np
from scipy.interpolate import interp1d

from pycbc.types import TimeSeries
from pycbc.waveform.utils import taper_timeseries
try:
    from nr_eob_ub.sim.NRsim import NRsim
except ModuleNotFoundError:
    print('ICCUB NR sims module not found')

import spherical
import quaternionic

pi = np.pi
MTSUN_SI = 4.925491025543576e-06 # mass of Sun in seconds
MRSUN_SI = 1476.6250614046494 # mass of Sun in meters
MPC_SI = 1e6*3.085677581491367e+16 # Megaparsec in meters

class NRInjections():
    def __init__(self, tlen, sample_rate):
        """
        Initialize the injection class
        
        tlen: time length of analysis segment in seconds
            Should be longer than longest waveform
            
        sample_rate: sample rate in Hz
            Should be higher than twice the highest frequency in the waveform
        """

        if not isinstance(tlen, int) or not isinstance(sample_rate, int):
            raise ValueError('tlen and sample_rate must be integers.')
        self.tlen = tlen
        self.srate = sample_rate
        self.Nsamp = tlen*sample_rate
        self.wigner = spherical.Wigner(8)
        self.modes = [(2,2), (2,-2)]
    
    def set_modes(self, modes = [(2,2), (2,-2)]):
        self.modes = modes

    def load_nr_sim(self, path, name):
        return NRsim(path, name, lm = self.modes, method='ffi')
    
    def _nr_waveform(self, nrsim, theta, phi):
        """Internal function to make the unscaled NR waveform"""
        R = quaternionic.array.from_spherical_coordinates(theta, phi)
        Y = self.wigner.sYlm(-2,R)
        output = 0.
        for mode in self.modes:
            try:
                thismode = nrsim.get_mode(mode, tp = 'h').value
            except KeyError:
                thismode = nrsim.get_mode((mode[0], -mode[1]), tp = 'h').value.conjugate()
            output += thismode * Y[self.wigner.Yindex(*mode)]
        time = (nrsim.get_mode(self.modes[0], tp='h')).time
        return np.array((time, output.real, -output.imag)).T
    
    def asc_waveform(self, nrsim, mtotal, theta, phi, distance):
        try:
            data = self._nr_waveform(nrsim, theta, phi)
        except AttributeError:
            data = nrsim # Allow direct setting of data for testing
        # Decide what total mass (in Msol) you want
        # Then scale t/M to get physical time in seconds
        mfac = MTSUN_SI*mtotal
        times = mfac*data[:,0]
        times -= times[0]
        endtime = times[-1]
        delta_t = 1/self.srate

        sample_times = np.arange(0, endtime, delta_t)
        func1 = interp1d(times, data[:,1])

        scale = (MRSUN_SI*mtotal)/(MPC_SI*distance)
        hp = TimeSeries(scale*func1(sample_times), delta_t=delta_t)
        hp.resize(self.Nsamp)
        idx = np.abs(hp.data).argmax()
        return hp.cyclic_time_shift(self.tlen-1.-idx/self.srate)
    def h5_waveform(self, h5f, mtotal, theta, phi, distance):
        mfac = MTSUN_SI*mtotal
        delta_t = 1/self.srate
        R = quaternionic.array.from_spherical_coordinates(theta, phi)
        Y = self.wigner.sYlm(-2,R)
        scale = (MRSUN_SI*mtotal)/(MPC_SI*distance)
        
        times = np.array(h5f['NRTimes'])
        X0 = times[0]
        endtime = mfac*(times[-1] - X0)
        
        sample_times = np.arange(0, endtime, delta_t)
        def interp(data):
            x = mfac*(np.array(data['X']) - X0)
            y = np.array(data['Y'])
            return interp1d(x, y, bounds_error=False, fill_value=0.)(sample_times)
        def mode_interp(ell, m):
            amp = interp(h5f[f"amp_l{ell}_m{m}"])
            phase = interp(h5f[f"phase_l{ell}_m{m}"])
            return amp*np.exp(1.j*phase)

        output = 0.
        for mode in self.modes:
            output += mode_interp(*mode) * Y[self.wigner.Yindex(*mode)]
        idx = np.abs(output).argmax()
        
        hp = TimeSeries(scale*output.real, delta_t=delta_t)
        hp.resize(self.Nsamp)
        return hp.cyclic_time_shift(self.tlen-1.-delta_t*idx)