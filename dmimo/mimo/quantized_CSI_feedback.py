import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sionna

class quantized_CSI_feedback(Layer):
    """CSI feedback report generation"""

    def __init__(self,
                method,
                dtype=tf.complex64,
                **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        
        self.method = method
        

    def call(self, h_est):
        
        if self.method == '5G':
            
            PMI = self.cal_PMI(h_est)
            CQI = NULL
            RI = NULL

            CSI_feedback_report = [PMI, CQI, RI]
        elif self.method == 'RVQ':

            CSI_feedback_report = cal_RVQ_CSI(h_est)
        else:
            raise Exception(f"The {self.method} CSI feedback mechanism has not been implemented. The simulator supports 5G standard CSI feedback and RVQ CSI feedback only.")
        
        return CSI_feedback_report

    def cal_PMI(self, h_est):

        N_t = h_est.shape[4]
        N_r = h_est.shape[2]

        if N_t == 4 and N_r == 2:

            PMI = 1

        elif N_t == 4 and N_r == 4:

            PMI = 2

        else:
            raise Exception(f"5G standard PMI feedback for {N_t} x {N_r} MIMO order has not been implemented. The simulator supports MIMO orders 4x2 and 4x4 only.")



