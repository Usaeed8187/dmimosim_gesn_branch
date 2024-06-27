import tensorflow as tf
from tensorflow.keras.layers import Layer
import sionna
from sionna.utils import flatten_dims
from sionna.ofdm import RemoveNulledSubcarriers

from .precoding import sumimo_svd_equalizer


class SVDEqualizer(Layer):

    def __init__(self,
                 resource_grid,
                 stream_management,
                 return_effective_channel=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        assert isinstance(resource_grid, sionna.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.mimo.StreamManagement)
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._return_effective_channel = return_effective_channel
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

    def call(self, inputs):

        y, h = inputs
        # y has shape
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        #
        # h has shape
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols,...
        # ..., fft_size]

        # Transformations to bring h and y in the desired shapes

        # Transpose y:
        # [batch_size, num_rx, num_ofdm_symbols, fft_size, num_streams_per_tx]
        y_equalized = tf.transpose(y, [0, 1, 3, 4, 2])
        y_equalized = tf.cast(y_equalized, self._dtype)

        # Transpose h:
        # [num_tx, num_rx, num_rx_ant, num_tx_ant, num_ofdm_symbols,...
        #  ..., fft_size, batch_size]
        h_eq = tf.transpose(h, [3, 1, 2, 4, 5, 6, 0])

        # Gather desired channel for precoding:
        # [num_tx, num_rx_per_tx, num_rx_ant, num_tx_ant, num_ofdm_symbols,...
        #  ..., fft_size, batch_size]
        h_eq_desired = tf.gather(h_eq, self._stream_management.precoding_ind,
                                 axis=1, batch_dims=1)

        # Flatten dims 2,3:
        # [num_tx, num_rx_per_tx * num_rx_ant, num_tx_ant, num_ofdm_symbols,...
        #  ..., fft_size, batch_size]
        h_eq_desired = flatten_dims(h_eq_desired, 2, axis=1)

        # Transpose:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size,...
        #  ..., num_streams_per_tx, num_tx_ant]
        h_eq_desired = tf.transpose(h_eq_desired, [5, 0, 3, 4, 1, 2])
        h_eq_desired = tf.cast(h_eq_desired, self._dtype)

        # SVD precoding
        y_equalized = sumimo_svd_equalizer(y_equalized, h_eq_desired)

        # Transpose output to desired shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        y_equalized = tf.transpose(y_equalized, [0, 1, 4, 2, 3])

        return y_equalized
