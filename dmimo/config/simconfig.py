# Configuration for system simulation

from .sysconfig import CarrierConfig


class SimConfig(CarrierConfig):

    def __init__(self, **kwargs):
        self._name = "Simulation Configuration"
        self._num_tx_ue_sel = 8                 # number of Tx UE selected
        self._num_rx_ue_sel = 8                 # number of Rx UE selected
        self._modulation_order = 2              # modulation order for non-adaptive case
        self._code_rate = 0.5                   # LDPC code rate
        self._num_tx_streams = 2                # total number of transmitter streams
        self._start_slot_idx = 15               # start slot index for simulation
        self._csi_delay = 2                     # CSI estimation delay
        self._first_slot_idx = 0                # first slot index for phase 2 in simulation
        self._num_slots_p1 = 1                  # number of slots in phase 1/3
        self._num_slots_p2 = 3                  # number of slots in phase 2
        self._total_slots = 20                  # total slots of ns-3 channels
        self._ns3_folder = "../ns3/channels"    # data folder for ns-3 channels
        self._precoding_method = "ZF"           # precoding method
        self._ue_indices = None                 # UE antennas indices for precoding
        self._ue_ranks = None                   # UE ranks for precoding
        self._perfect_csi = False               # Use perfect CSI for debugging
        self._csi_prediction = False            # Use CSI prediction
        self._sto_sigma = 0.0                   # standard deviation of STO in nanoseconds
        self._cfo_sigma = 0.0                   # standard deviation of CFO in Hz
        self._rank_adapt = True                 # turn on rank adaptation
        self._link_adapt = True                 # turn on link adaptation
        self._return_estimated_channel = False  # return estimated channel without doing any symbol detection
        super().__init__(**kwargs)

    @property
    def num_tx_ue_sel(self):
        return self._num_tx_ue_sel

    @num_tx_ue_sel.setter
    def num_tx_ue_sel(self, val):
        self._num_tx_ue_sel = val

    @property
    def num_rx_ue_sel(self):
        return self._num_rx_ue_sel

    @num_rx_ue_sel.setter
    def num_rx_ue_sel(self, val):
        self._num_rx_ue_sel = val

    @property
    def modulation_order(self):
        return self._modulation_order

    @modulation_order.setter
    def modulation_order(self, val):
        self._modulation_order = val

    @property
    def code_rate(self):
        return self._code_rate

    @code_rate.setter
    def code_rate(self, val):
        self._code_rate = val

    @property
    def num_tx_streams(self):
        return self._num_tx_streams

    @num_tx_streams.setter
    def num_tx_streams(self, val):
        self._num_tx_streams = val

    @property
    def start_slot_idx(self):
        return self._start_slot_idx

    @start_slot_idx.setter
    def start_slot_idx(self, val):
        self._start_slot_idx = val

    @property
    def csi_delay(self):
        return self._csi_delay

    @csi_delay.setter
    def csi_delay(self, val):
        self._csi_delay = val

    @property
    def first_slot_idx(self):
        return self._first_slot_idx

    @first_slot_idx.setter
    def first_slot_idx(self, val):
        self._first_slot_idx = val

    @property
    def num_slots_p1(self):
        return self._num_slots_p1

    @num_slots_p1.setter
    def num_slots_p1(self, val):
        self._num_slots_p1 = val

    @property
    def num_slots_p2(self):
        return self._num_slots_p2

    @num_slots_p2.setter
    def num_slots_p2(self, val):
        self._num_slots_p2 = val

    @property
    def total_slots(self):
        return self._total_slots

    @total_slots.setter
    def total_slots(self, val):
        self._total_slots = val

    @property
    def ns3_folder(self):
        return self._ns3_folder

    @ns3_folder.setter
    def ns3_folder(self, val):
        self._ns3_folder = val

    @property
    def precoding_method(self):
        return self._precoding_method

    @precoding_method.setter
    def precoding_method(self, val):
        self._precoding_method = val

    @property
    def ue_indices(self):
        return self._ue_indices

    @ue_indices.setter
    def ue_indices(self, val):
        self._ue_indices = val

    @property
    def ue_ranks(self):
        return self._ue_ranks

    @ue_ranks.setter
    def ue_ranks(self, val):
        self._ue_ranks = val

    @property
    def perfect_csi(self):
        return self._perfect_csi

    @perfect_csi.setter
    def perfect_csi(self, val):
        self._perfect_csi = val

    @property
    def csi_prediction(self):
        return self._csi_prediction

    @csi_prediction.setter
    def csi_prediction(self, val):
        self._csi_prediction = val

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
    
    @property
    def rank_adapt(self):
        return self._rank_adapt

    @rank_adapt.setter
    def rank_adapt(self, val):
        self._rank_adapt = val
    
    @property
    def link_adapt(self):
        return self._link_adapt

    @link_adapt.setter
    def link_adapt(self, val):
        self._link_adapt = val

    @property
    def return_estimated_channel(self):
        return self._return_estimated_channel

    @return_estimated_channel.setter
    def return_estimated_channel(self, val):
        self._return_estimated_channel = val
