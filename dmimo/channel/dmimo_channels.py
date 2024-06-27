"""
Layer for implementing an dMIMO channels in the frequency domain,
including TxSquad, dMIMO, and RxSquad models
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from sionna.channel import ApplyOFDMChannel, AWGN

from sionna.ofdm import ResourceGrid

from dmimo.config import Ns3Config
from .ns3_channels import LoadNs3Channel


class dMIMOChannels(Layer):
    # pylint: disable=line-too-long

    def __init__(self, config: Ns3Config, channel_type, resource_grid: ResourceGrid=None,
                 add_noise=True, normalize_channel=False, return_channel=False,
                 dtype=tf.complex64, **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)

        self._config = config
        self._channel_type = channel_type
        self._rg = resource_grid
        self._add_noise = add_noise
        self._normalize_channel = normalize_channel
        self._return_channel = return_channel
        self._load_channel = LoadNs3Channel(self._config)
        self._apply_channel = ApplyOFDMChannel(add_awgn=False, dtype=tf.as_dtype(self.dtype))
        self._awgn = AWGN(dtype=dtype)

    def call(self, inputs):

        # x: channel input samples, sidx: current slot index
        x, sidx = inputs

        # x has shape [batch_size, num_tx, num_tx_ant, num_ofdm_sym, fft_size]
        batch_size = tf.shape(x)[0]
        total_tx_ant = tf.shape(x)[1] * tf.shape(x)[2]
        # num_txs_ant = self._config.num_bs * self._config.num_bs_ant + self._config.num_txue * self._config.num_ue_ant
        # assert num_txs_ant == total_tx_ant, "Total number of transmit antennas of input and channel must match"

        # load pre-generated channel
        # h_freq shape: [batch_size, num_rx_ant, num_tx_ant, num_ofdm_sym, fft_size]
        # snrdb shape:
        h_freq, snrdb = self._load_channel(self._channel_type, slot_idx=sidx, batch_size=batch_size)

        # TODO: Tx/Rx UE selection, Resource Grid handling

        # prune channel coefficients if necessary
        if self._rg and x.shape[-1] != h_freq.shape[-1]:
            scidx = self._rg.effective_subcarrier_ind
            if x.shape[-1] != self._rg.num_effective_subcarriers:
                x = tf.gather(x, scidx, axis=-1)
            if h_freq.shape[-1] != self._rg.num_effective_subcarriers:
                h_freq = tf.gather(h_freq, scidx, axis=-1)

        # apply channel to inputs
        y = self._apply_channel([x, h_freq])  # [batch_size, num_rx, num_rx_ant, num_ofdm_sym, fft_size]

        # Add thermal noise
        if self._add_noise:
            no = np.power(10.0, snrdb / (-10.0))
            no = np.expand_dims(no, -1)  # [batch_size, num_rx, num_rx_ant, num_ofdm_sym, 1]
            y = self._awgn([y, no])

        if self._return_channel:
            return y, h_freq
        else:
            return y
