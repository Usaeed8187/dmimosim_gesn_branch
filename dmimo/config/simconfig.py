# Configuration for system simulation

from .config import Config


class SimConfig(Config):

    def __init__(self, **kwargs):
        self._name = "Simulation Configuration"
        self._fft_size = 512                # FFT size
        self._subcarrier_spacing = 15e3     # subcarrier spacing in Hz
        self._sto_sigma = 0.0               # standard deviation of STO in nanoseconds
        self._cfo_sigma = 0.0               # standard deviation of CFO in Hz

        super().__init__(**kwargs)

    @property
    def fft_size(self):
        return self._fft_size

    @fft_size.setter
    def fft_size(self, val):
        assert 0 < val <= 4096, "Invalid FFT size"
        self._fft_size = val

    @property
    def subcarrier_spacing(self):
        return self._subcarrier_spacing

    @subcarrier_spacing.setter
    def subcarrier_spacing(self, val):
        self._subcarrier_spacing = val

    @property
    def sto_sigma(self):
        return self._sto_sigma

    @sto_sigma.setter
    def sto_sigma(self, val):
        self._sto_sigma = val

    @property
    def cfo_sigma(self):
        return self._cfo_sigma

    @cfo_sigma.setter
    def cfo_sigma(self, val):
        self._cfo_sigma = val

