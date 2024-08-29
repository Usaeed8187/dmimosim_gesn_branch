## dMIMO Python System Simulator Modules

## Overview
The core simulator library in the "**dmimo**" folder contains all the algorithms
and modules for the dMIMO operations, including coherent joint transmission (CJT)
for SU/MU-MIMO scenarios, non-coherent (NCJT), and Tx/Rx squad transmission.

The "**sims**" folder contains the top-level simulation scripts for baseline and 
SU/MU-MIMO scenarios. The corresponding simulation results will be saved
in the "**results**" folder.

Some standard signal processing components from [Sionna](https://nvlabs.github.io/sionna/) 
are used in the simulation, such as OFDM modulation/demodulation, resource grid, 
pilot pattens, MIMO equalization/detection, and LDPC encoding/decoding. 
Please refer to Sionna's [documentation](https://nvlabs.github.io/sionna/api/sionna.html) 
for these components.
The "**dmimo**" library follows the Sionna's conventions in tensor signal processing. 

## Core simulator library

To use the "**dmimo**" library, make sure to include the "dmimo" folder in the 
Python system path. See the example usage in the simulation scripts 
in the "sims" folder.

### Configuration

The following configuration are used for system and simulation settings.
To use different settings, change their values in the top-simulation scripts. 
1) NetworkConfig: dMIMO network architecture settings
2) CarrierConfig: RF carrier settings
3) MCSConfig: Modulation and coding schemes
4) Ns3Config: Configuration for the ns-3 simulation
5) SimConfig: Settings for simulation runs


### dMIMO Scenarios

The simulation model for baseline and SU-MIMO scenarios are implemented, 
together with the Tx squad transmission. 

##### 1. Baseline (baseline.py)

Simulation of baseline scenarios using 4x4 MIMO channels between Tx squad gBN
and Rx squad gNB.
It loads the ns-3 channels using the "Baseline" channel type.

##### 2. SU_MIMO (su_mimo.py)

Signal processing for one SU-MIMO transmission cycle.  
Effective channel models for phase 2 (P2) and phase 3 (P3) are used,
phase 1 (P1) transmission is assumed to always provide enough 
data bandwidth for P2.

##### 3. TxSquad (txs_mimo.py)

Signal processing for TxSquad downlink transmission in phase 1 (P1).
Currently, it supports downlink bandwidth up to 150 Mbps.  Higher 
bandwidth requirements and improved implementation will be added in
future releases.

### dMIMO channel modules

##### 1. LoadNs3Channel
The LoadNs3Channel module loads and convert pre-generated channels from the ns-3 simulation
for specified channel types and slot/subframe indices. It will return the effective channels
combining small scale fading and large-scale fading for each OFDM symbols 
in specified slots/subframes.

The channel types are listed below.
1) Baseline: direct gNB-to-gBN 4x4 MIMO channels for comparison purpose.
2) TxSquad: intra-squad channels between Tx gBN and Tx squad UEs.
3) RxSquad: intra-squad channels between Rx gNB and Rx squad UEs.
4) dMIMO: mobile MIMO channels between Tx/Rx squads (including gNB and UEs), 
   used in phase 2 operation for MU-MIMO scenario.
5) dMIMO-Forward: effective MIMO channels with analog forwarding for phase 3, 
   used in phase 2 and 3 operations for SU-MIMO scenario.  
6) dMIMO-Raw: small scaling fading and pathloss for the mobile MIMO channels (phase 2) 
   without combining them, for NCJT scenario and other tests. 

##### 2. dMIMOChannels

Wrapper class to apply dMIMO channels in simulation scripts.  It handles passing
transmitter signals through the dMIMO channels and adding AWGN noise.

##### 3. Channel estimation functions

1) **estimate_freq_cov**: Estimate frequency-domain covariance 
   of the given MIMO channels.
2) **estimate_freq_time_cov***: Estimate both frequency-domain and time-domain covariance 
   of the given MIMO channels.
3) **lmmse_channel_estimation**: Estimate the frequency-domain MIMO channels for given 
  dMIMO channel types. Frequency-domain LMMSE interpolation (LMMSEInterpolator1D) and 
  time-domain linear interpolation (LinearInterp1D) are used.


### MIMO Precoding/equalization

##### 1. SVDPrecoder and SVDEqualizer

Precoder and equalizer modules using SVD methods for the SU-MIMO scenario, for reference and
comparison purpose.

The functions **sumimo_svd_precoder** and **sumimo_svd_equalizer** provide 
the implementations of the algorithms.


##### 2. ZFPrecoder

Precoder modules using zero-forcing methods for the SU/MU-MIMO scenario.

The functions **sumimo_zf_precoder** and **mumimo_zf_precoder** provide 
the implementations of the algorithms.


