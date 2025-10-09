import tensorflow as tf

def rb_to_subcarriers(h_rb, fft_size=512, method="lanczos3"):
    
    """
    h_rb: complex tensor with frequency as the last dim (length = #RBs)
    
    h_rb: [..., F] complex64/complex128 (frequency last)
    returns: [..., fft_size]
    """
    # Collapse leading dims, resize as an "image" of height=1, width=F, channels=1
    x = tf.convert_to_tensor(h_rb)
    shape = tf.shape(x)
    F = shape[-1]
    flat = tf.reshape(x, [-1, F])

    # Option A: interpolate real/imag separately
    xr = tf.math.real(flat)[..., tf.newaxis]  # [N, F, 1]
    xi = tf.math.imag(flat)[..., tf.newaxis]

    # tf.image.resize expects [N, H, W, C]; use H=1, W=F
    xr = tf.expand_dims(xr, 1)  # [N,1,F,1]
    xi = tf.expand_dims(xi, 1)

    xr_i = tf.image.resize(xr, size=[1, fft_size], method=method)  # [N,1,new_len,1]
    xi_i = tf.image.resize(xi, size=[1, fft_size], method=method)

    xr_i = tf.squeeze(xr_i, axis=[1,3])  # [N, new_len]
    xi_i = tf.squeeze(xi_i, axis=[1,3])

    out = tf.complex(xr_i, xi_i)         # [N, new_len]
    out = tf.reshape(out, tf.concat([shape[:-1], [fft_size]], axis=0))
    return out