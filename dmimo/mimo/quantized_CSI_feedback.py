import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sionna

class quantized_CSI_feedback(Layer):
    """CSI feedback report generation"""

    def __init__(self,
                method,
                num_tx_streams,
                dtype=tf.complex64,
                **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        
        self.method = method
        if self.method == '5G':
            self.N_1 = 2 # Number of quantization points in the horizontal dimension
            self.N_2 = 1 # Number of quantization points in the vertical dimension
            self.O_1 = 4 # Horizontal oversampling factor
            self.O_2 = 1 # Vertical oversampling factor
            self.num_tx_streams = num_tx_streams
        

    def call(self, h_est):
        
        if self.method == '5G':

            codebook = self.cal_codebook(h_est)
            PMI = self.cal_PMI(codebook)
            CQI = None
            RI = None

            CSI_feedback_report = [PMI, CQI, RI]
        elif self.method == 'RVQ':

            CSI_feedback_report = cal_RVQ_CSI(h_est)
        else:
            raise Exception(f"The {self.method} CSI feedback mechanism has not been implemented. The simulator supports 5G standard CSI feedback and RVQ CSI feedback only.")
        
        return CSI_feedback_report

    def cal_codebook(self, h_est):
        """
        Computes PMI for 4x2 and 4x4 MIMO configuration (format N_t x N_r)
        Consult 3GPP TS 38.214 Section 5 for details
        """

        N_t = h_est.shape[4]
        N_r = h_est.shape[2]
        P_CSI_RS = N_t

        if N_t == 4 and N_r == 2:
            
            if self.num_tx_streams == 1:
                
                i_11 = np.arange(0, self.N_1 * self.O_1)
                i_12 = np.arange(0, self.N_2 * self.O_2)
                i_2 = np.arange(0,4)

                # u_m = self.compute_u_m(i_12)
                # # v_l_m = 
                # # W_i11_i12_i2 = 
                # W = 1/np.sqrt(P_CSI_RS) * W


        elif N_t == 4 and N_r == 4:

            if self.num_tx_streams == 1:
                
                i_11 = np.arange(0, self.N_1 * self.O_1)
                i_12 = np.arange(0, self.N_2 * self.O_2)
                i_2 = np.arange(0,4)

                l_all = i_11
                m_all = i_12
                n_all = i_2

                W_lmn = np.zeros((len(l_all), len(m_all), len(n_all), N_t, self.num_tx_streams), dtype=complex)

                for l in l_all:
                    for m in m_all:

                        v_lm = self.compute_v_lm(l, m)
                        
                        for n in n_all:

                            phi_n = np.exp(1j * np.pi * n / 2)

                            W_lmn[l,m,n,...] = np.vstack((v_lm, phi_n * v_lm))

                W = 1/np.sqrt(P_CSI_RS) * W_lmn

            elif self.num_tx_streams == 2:

                i_11 = np.arange(0, self.N_1 * self.O_1)
                i_12 = np.arange(0, self.N_2 * self.O_2)
                i_13 = np.arange(0,2)
                i_2 = np.arange(0,2)

                l_all = i_11
                m_all = i_12
                n_all = i_2

        else:
            raise Exception(f"5G standard PMI feedback for {N_t} x {N_r} MIMO order has not been implemented. The simulator supports MIMO orders 4x2 and 4x4 only.")
        
        return W


    def compute_u_m(self, m):

        if self.N_2 == 1:
            u_m = 1
        elif self.N_2 > 1:
            u_m = np.exp((2j * np.pi * m * np.arange(0, self.N_2)) / (self.O_2 * self.N_2) )
        else:
            raise Exception(f"Incorrect choice of N_2")

        return u_m

    def compute_v_lm(self, l, m):

        u_m = self.compute_u_m(m)

        v_l = np.exp( (2j * np.pi * l * np.arange(0, self.N_1)) / (self.O_1 * self.N_1))

        v_l_m = np.outer(u_m, v_l).flatten()
        v_l_m = v_l_m.reshape(-1, 1)

        return v_l_m
    
    def cal_PMI(self, codebook):
        
        PMI = 1

        return PMI

