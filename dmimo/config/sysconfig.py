# dMIMO network scenarios

from .config import Config


class NetworkConfig(Config):

    def __init__(self, **kwargs):
        self._name = "Network Configuration"
        self._num_bs = 1            # number of basestation per squad, always 1
        self._num_txue = 10         # number of transmit squad UEs
        self._num_rxue = 10         # number receiving squad UEs
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
        self._num_txue = val

    @property
    def num_rxue(self):
        return self._num_rxue

    @num_rxue.setter
    def num_rxue(self, val):
        self._num_rxue = val

    @property
    def num_bs_ant(self):
        return self._num_bs_ant

    @num_bs_ant.setter
    def num_bs_ant(self, val):
        self._num_bs_ant = val

    @property
    def num_ue_ant(self):
        return self._num_ue_ant

    @num_ue_ant.setter
    def num_ue_ant(self, val):
        self._num_ue_ant = val

    @property
    def txue_mask(self):
        return self._txue_mask

    @txue_mask.setter
    def txue_mask(self, val):
        assert isinstance(val, list)
        assert len(val) == self._num_txue
        self._txue_mask = val

    @property
    def rxue_mask(self):
        return self._rxue_mask

    @rxue_mask.setter
    def rxue_mask(self, val):
        assert isinstance(val, list)
        assert len(val) == self._num_rxue
        self._rxue_mask = val


