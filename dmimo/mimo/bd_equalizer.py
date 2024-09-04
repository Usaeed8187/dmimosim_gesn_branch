import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sionna
from sionna.utils import flatten_dims
from sionna.ofdm import RemoveNulledSubcarriers

from .bd_precoding import mumimo_bd_equalizer


class BDEqualizer(Layer):
    """BD Equalizer for MU-MIMO"""

    def __init__(self,
                 resource_grid,
                 stream_management,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        assert isinstance(resource_grid, sionna.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.mimo.StreamManagement)
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

    def call(self, inputs):

        ue_rank_adapt = False
        if len(inputs) == 2:
            # all user has the same number of streams/antennas
            y, h = inputs
        elif len(inputs) == 4:
            # specify user Rx antennas indices and streams (rank)
            y, h, ue_indices, ue_ranks = inputs
            if ue_indices is not None and ue_ranks is not None:
                ue_rank_adapt = True
                if np.size(np.array(ue_ranks)) == 1:
                    ue_ranks = np.repeat(ue_ranks, len(ue_indices), axis=0)
        else:
            ValueError("calling BD precoder with incorrect params")

        # y has shape
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        #
        # h has shape
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]

        # Transformations to bring h and y in the desired shapes

        # Transpose y:
        # [batch_size, num_rx, num_ofdm_symbols, fft_size, num_streams_per_tx]
        y_equalized = tf.transpose(y, [0, 1, 3, 4, 2])
        y_equalized = tf.cast(y_equalized, self._dtype)

        # Transpose h:
        # [num_tx, num_rx, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        h_eq = tf.transpose(h, [3, 1, 2, 4, 5, 6, 0])

        # Gather desired channel for precoding:
        # [num_tx, num_rx_per_tx, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        h_eq_desired = tf.gather(h_eq, self._stream_management.precoding_ind,
                                 axis=1, batch_dims=1)

        # Flatten dims 2,3:
        # [num_tx, num_rx_per_tx * num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        h_eq_desired = flatten_dims(h_eq_desired, 2, axis=1)

        # Transpose:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx, num_tx_ant]
        h_eq_desired = tf.transpose(h_eq_desired, [5, 0, 3, 4, 1, 2])
        h_eq_desired = tf.cast(h_eq_desired, self._dtype)

        # Rx antenna indices for MU-MIMO
        if ue_rank_adapt is False:
            # all user has the same number of antennas
            # no rank adaptation for all users
            num_ue, num_ue_ant = h_eq.shape[1:3]
            ue_ranks = np.repeat([num_ue_ant], num_ue, axis=0)
            ue_indices = []
            for k in range(num_ue):
                offset = num_ue_ant * k  # first antennas index for k-th UE
                ue_indices.append(np.arange(offset, offset + num_ue_ant))
        else:
            # check rx_indices and rx_ranks
            num_rx_ant = [len(val) for val in ue_indices]
            total_rx_ant = np.sum(num_rx_ant)
            assert total_rx_ant == h_eq_desired.shape[4], "total number of UE antennas must match channel coefficients"
            assert all(ue_ranks <= num_rx_ant), "UE rank should not exceed number of antennas"

        # BD equalizing
        y_equalized = mumimo_bd_equalizer(y_equalized, h_eq_desired, ue_indices, ue_ranks)

        # Transpose output to desired shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        y_equalized = tf.transpose(y_equalized, [0, 1, 4, 2, 3])

        return y_equalized
