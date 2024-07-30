# dMIMO network scenarios
import numpy as np

from .config import Config


class NetworkConfig(Config):

    def __init__(self, **kwargs):
        self._name = "Network Configuration"
        self._num_bs = 1            # number of basestation per squad, always 1
        self._num_txue = 10         # number of transmit squad UEs
        self._num_rxue = 10         # number of receiving squad UEs
        self._num_bs_ant = 4        # number of antennas per BS
        self._num_ue_ant = 2        # number of antennas per UE
        self._txue_mask = None      # selection mask for transmitting squad UEs
        self._rxue_mask = None      # selection mask for receiving squad UEs
        super().__init__(**kwargs)

    @property
    def num_bs(self):
        return self._num_bs

    @property
    def num_txue(self):
        return self._num_txue

    @num_txue.setter
    def num_txue(self, val):
        assert 0 < val <= 20, "Invalid number of Tx UEs"
        self._num_txue = val

    @property
    def num_rxue(self):
        return self._num_rxue

    @num_rxue.setter
    def num_rxue(self, val):
        assert 0 < val <= 20, "Invalid number of Rx UEs"
        self._num_rxue = val

    @property
    def num_bs_ant(self):
        return self._num_bs_ant

    @num_bs_ant.setter
    def num_bs_ant(self, val):
        assert 0 < val <= 8, "Invalid number of BS antennas"
        self._num_bs_ant = val

    @property
    def num_ue_ant(self):
        return self._num_ue_ant

    @num_ue_ant.setter
    def num_ue_ant(self, val):
        assert 0 < val <= 4, "Invalid number of UE antennas"
        self._num_ue_ant = val

    @property
    def txue_mask(self):
        return self._txue_mask

    @txue_mask.setter
    def txue_mask(self, val):
        assert isinstance(val, list) or isinstance(val, np.ndarray), "Invalid Tx UE selection mask"
        val = np.reshape(val, -1)
        self._txue_mask = (val != 0)

    @property
    def rxue_mask(self):
        return self._rxue_mask

    @rxue_mask.setter
    def rxue_mask(self, val):
        assert isinstance(val, list) or isinstance(val, np.ndarray), "Invalid Rx UE selection mask"
        val = np.reshape(val, -1)
        self._rxue_mask = (val != 0)


class CarrierConfig(Config):

    def __init__(self, **kwargs):
        self._name = "Carrier Configuration"
        self._fft_size = 512                # FFT size
        self._cyclic_prefix_len = 64        # cyclic prefix length
        self._subcarrier_spacing = 15e3     # subcarrier spacing in Hz
        self._slot_duration = 1e-3          # slot duration in seconds
        super().__init__(**kwargs)

    @property
    def fft_size(self):
        return self._fft_size

    @fft_size.setter
    def fft_size(self, val):
        assert 0 < val <= 4096, "Invalid FFT size"
        self._fft_size = val

    @property
    def cyclic_prefix_len(self):
        return self._cyclic_prefix_len

    @cyclic_prefix_len.setter
    def cyclic_prefix_len(self, val):
        assert 0 < val <= 1024, "Invalid cyclic prefix length"
        self._cyclic_prefix_len = val

    @property
    def subcarrier_spacing(self):
        return self._subcarrier_spacing

    @subcarrier_spacing.setter
    def subcarrier_spacing(self, val):
        self._subcarrier_spacing = val

    @property
    def slot_duration(self):
        return self._slot_duration

    @slot_duration.setter
    def slot_duration(self, val):
        self._slot_duration = val
