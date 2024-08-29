# Configuration for NCJT simulation

from .sysconfig import CarrierConfig


class NcjtSimConfig(CarrierConfig):

    def __init__(self, **kwargs):
        self._name = "NCJT Simulation Configuration"
        self._ns3_folder = "../ns3/channels"  # data folder for ns-3 channels
        self._EQUALIZER = 100  # Scales the received power and noise power by the same amount in dB scale to ensure floating point stability
        self._NOISE_FLOOR = -105  # Noise floor power in dBm
        self._NOISE_FIGURE = 5  # Noise figure in dB
        self._ANTENNA_GAIN = 5  # Combined transceiver antenna gain in dB
        self._UeTxPwrdB = 26  # Each UE transmit power in dBm
        self._BsTxPwrdB = 34  # Transmit Base Station power in dBm
        self._num_subframes = 10  # Number of subframes
        self._starting_subframe = 2  # Index of the starting subframe
        self._num_subframes_phase1 = 3  # number of subframes in the Tx Squad phase
        self._num_subframes_phase2 = 6  # number of subframes in the dMIMO phase
        self._num_ofdm_symbols = 14  # Number of OFDM symbols in each subframe
        self._num_subcarriers = 512  # Number of subcarriers
        self._num_TxBs = 1  # Number of transmit base stations participating
        self._num_RxBs = 1  # Number of receive base stations participating
        self._num_TxUe = 10  # Number of transmit UEs participating
        self._num_RxUe = 10  # Number of receive UEs participating
        self._nAntTxUe = 2  # Number of transmit antennas of the Tx UEs
        self._nAntTxBs = 4  # Number of transmit antennas of the Tx BS
        self._nAntRxUe = 2  # Number of transmit antennas of the Rx UEs
        self._nAntRxBs = 4  # Number of transmit antennas of the Rx BS
        self._num_bits_per_symbol_phase2 = 2  # 2 for QPSk and 4 for 16 QAM
        self._num_bits_per_symbol_phase1 = 6  # 2 for QPSk and 4 for 16 QAM
        self._num_bits_per_symbol_phase3 = 6  # 2 for QPSk and 4 for 16 QAM
        self._perSC_SNR = False  # Whether per subcarrier SNR is available at the Rx BS for post detection fusion

        super().__init__(**kwargs)

    @property
    def ns3_folder(self):
        return self._ns3_folder

    @ns3_folder.setter
    def ns3_folder(self, val):
        self._ns3_folder = val

    @property
    def EQUALIZER(self):
        return self._EQUALIZER

    @EQUALIZER.setter
    def EQUALIZER(self, val):
        self._EQUALIZER = val

    @property
    def NOISE_FLOOR(self):
        return self._NOISE_FLOOR

    @NOISE_FLOOR.setter
    def NOISE_FLOOR(self, val):
        self._NOISE_FLOOR = val

    @property
    def NOISE_FIGURE(self):
        return self._NOISE_FIGURE

    @NOISE_FIGURE.setter
    def NOISE_FIGURE(self, val):
        self._NOISE_FIGURE = val

    @property
    def ANTENNA_GAIN(self):
        return self._ANTENNA_GAIN

    @ANTENNA_GAIN.setter
    def ANTENNA_GAIN(self, val):
        self._ANTENNA_GAIN = val

    @property
    def UeTxPwrdB(self):
        return self._UeTxPwrdB

    @UeTxPwrdB.setter
    def UeTxPwrdB(self, val):
        self._UeTxPwrdB = val

    @property
    def BsTxPwrdB(self):
        return self._BsTxPwrdB

    @BsTxPwrdB.setter
    def BsTxPwrdB(self, val):
        self._BsTxPwrdB = val

    @property
    def BsTxPower(self):
        return self._BsTxPower

    @BsTxPower.setter
    def BsTxPower(self, val):
        self._BsTxPower = val

    @property
    def num_subframes(self):
        return self._num_subframes

    @num_subframes.setter
    def num_subframes(self, val):
        self._num_subframes = val

    @property
    def starting_subframe(self):
        return self._starting_subframe

    @starting_subframe.setter
    def starting_subframe(self, val):
        self._starting_subframe = val

    @property
    def num_subframes_phase1(self):
        return self._num_subframes_phase1

    @num_subframes_phase1.setter
    def num_subframes_phase1(self, val):
        self._num_subframes_phase1 = val

    @property
    def num_subframes_phase2(self):
        return self._num_subframes_phase2

    @num_subframes_phase2.setter
    def num_subframes_phase2(self, val):
        self._num_subframes_phase2 = val

    @property
    def num_ofdm_symbols(self):
        return self._num_ofdm_symbols

    @num_ofdm_symbols.setter
    def num_ofdm_symbols(self, val):
        self._num_ofdm_symbols = val

    @property
    def num_subcarriers(self):
        return self._num_subcarriers

    @num_subcarriers.setter
    def num_subcarriers(self, val):
        self._num_subcarriers = val

    @property
    def num_TxBs(self):
        return self._num_TxBs

    @num_TxBs.setter
    def num_TxBs(self, val):
        self._num_TxBs = val

    @property
    def num_RxBs(self):
        return self._num_RxBs

    @num_RxBs.setter
    def num_RxBs(self, val):
        self._num_RxBs = val

    @property
    def num_TxUe(self):
        return self._num_TxUe

    @num_TxUe.setter
    def num_TxUe(self, val):
        self._num_TxUe = val

    @property
    def num_RxUe(self):
        return self._num_RxUe

    @num_RxBs.setter
    def num_RxUe(self, val):
        self._num_RxUe = val

    @property
    def nAntTxUe(self):
        return self._nAntTxUe

    @nAntTxUe.setter
    def nAntTxUe(self, val):
        self._nAntTxUe = val

    @property
    def nAntTxBs(self):
        return self._nAntTxBs

    @nAntTxBs.setter
    def nAntTxBs(self, val):
        self._nAntTxBs = val

    @property
    def nAntRxUe(self):
        return self._nAntRxUe

    @nAntRxUe.setter
    def nAntRxUe(self, val):
        self._nAntRxUe = val

    @property
    def nAntRxBs(self):
        return self._nAntRxBs

    @nAntRxBs.setter
    def nAntRxBs(self, val):
        self._nAntRxBs = val

    @property
    def num_bits_per_symbol_phase2(self):
        return self._num_bits_per_symbol_phase2

    @num_bits_per_symbol_phase2.setter
    def num_bits_per_symbol_phase2(self, val):
        self._num_bits_per_symbol_phase2 = val

    @property
    def num_bits_per_symbol_phase1(self):
        return self._num_bits_per_symbol_phase1

    @num_bits_per_symbol_phase1.setter
    def num_bits_per_symbol_phase1(self, val):
        self._num_bits_per_symbol_phase1 = val

    @property
    def num_bits_per_symbol_phase3(self):
        return self._num_bits_per_symbol_phase3

    @num_bits_per_symbol_phase3.setter
    def num_bits_per_symbol_phase3(self, val):
        self._num_bits_per_symbol_phase3 = val

    @property
    def perSC_SNR(self):
        return self._perSC_SNR

    @perSC_SNR.setter
    def perSC_SNR(self, val):
        self._perSC_SNR = val

