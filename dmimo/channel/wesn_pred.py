import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, initializers, activations

def complex_initializer(real_init):
    """Create a complex initializer from a real-valued initializer."""

    def _init(shape, dtype=None):
        real = real_init(shape, dtype=tf.float32)
        imag = real_init(shape, dtype=tf.float32)
        return tf.complex(real, imag)

    return _init

class ESNCell(tf.keras.layers.Layer):
    """Echo State recurrent Network (ESN) cell.

    This cell implements an echo state network as described in:
        H. Jaeger,
        "The 'echo state' approach to analysing and training recurrent neural networks".
        GMD Report148, German National Research Center for Information Technology, 2001.
    """

    def __init__(
        self,
        units: int,
        connectivity: float = 0.1,
        leaky: float = 1.0,
        spectral_radius: float = 0.9,
        use_norm2: bool = False,
        use_bias: bool = True,
        activation=tf.math.tanh,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(dtype=tf.complex64, **kwargs)
        self.units = units
        self.connectivity = connectivity
        self.leaky = leaky
        self.spectral_radius = spectral_radius
        self.use_norm2 = use_norm2
        self.use_bias = use_bias
        # ``activation`` may be passed as a string or callable.  ``tf.math.tanh``
        # is used by default as it supports complex inputs.
        self.activation = activation if callable(activation) else activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    @property
    def state_size(self):
        # This tells the RNN wrapper the size of the state.
        return self.units

    @property
    def output_size(self):
        # This tells the RNN wrapper the size of the output.
        return self.units

    def build(self, inputs_shape):
        # Infer the last dimension of the input features.
        input_dim = inputs_shape[-1]
        if input_dim is None:
            raise ValueError("Input dimension must be defined in inputs_shape")

        # Define a custom recurrent initializer for the reservoir matrix.
        def _esn_recurrent_initializer(shape, dtype=None):
            # Initialize the recurrent weights.
            real = self.recurrent_initializer(shape, dtype=tf.float32)
            imag = self.recurrent_initializer(shape, dtype=tf.float32)
            recurrent_weights = tf.complex(real, imag)
            # Create a binary connectivity mask.
            connectivity_mask = tf.random.uniform(shape, dtype=tf.float32) <= self.connectivity
            recurrent_weights = tf.where(connectivity_mask, recurrent_weights, tf.zeros_like(recurrent_weights))

            # Scale the recurrent weights to satisfy the echo state property.
            if self.use_norm2:
                recurrent_norm2 = tf.norm(recurrent_weights, ord=2)
                scaling_factor = self.spectral_radius / (
                    tf.cast(recurrent_norm2, tf.float32)
                    + tf.cast(tf.equal(recurrent_norm2, 0), tf.float32)
                )
            else:
                abs_eig_values = tf.abs(tf.linalg.eigvals(recurrent_weights))
                max_abs_eig = tf.reduce_max(abs_eig_values)
                scaling_factor = tf.math.divide_no_nan(self.spectral_radius, max_abs_eig)
            return recurrent_weights * tf.cast(scaling_factor, tf.complex64)

        # Create the recurrent weight matrix (non-trainable).
        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=(self.units, self.units),
            initializer=_esn_recurrent_initializer,
            trainable=False,
            dtype=tf.complex64,
        )
        # Create the input kernel weight matrix (non-trainable).
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=complex_initializer(self.kernel_initializer),
            trainable=False,
            dtype=tf.complex64,
        )
        # Optionally, create the bias vector (non-trainable).
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=complex_initializer(self.bias_initializer),
                trainable=False,
                dtype=tf.complex64,
            )
        else:
            self.bias = None

        super().build(inputs_shape)

    @tf.function(jit_compile=True)
    def call(self, inputs, states):
        # Unpack previous state.
        prev_state = states[0]
        # Concatenate the input and previous state.
        in_matrix = tf.concat([inputs, prev_state], axis=-1)
        # Concatenate the input kernel and recurrent kernel.
        weights_matrix = tf.concat([self.kernel, tf.transpose(self.recurrent_kernel)], axis=0)
        # Compute the linear transformation.
        output = tf.matmul(in_matrix, weights_matrix)
        # Add bias if applicable.
        if self.use_bias:
            output += self.bias
        # Apply the activation function.
        output = self.activation(output)
        # Use leaky integration to combine previous state with the new output.
        leaky = tf.cast(self.leaky, output.dtype)
        output = (1 - leaky) * prev_state + leaky * output
        return output, [output]

    def get_config(self):
        config = {
            "units": self.units,
            "connectivity": self.connectivity,
            "leaky": self.leaky,
            "spectral_radius": self.spectral_radius,
            "use_norm2": self.use_norm2,
            "use_bias": self.use_bias,
            "activation": activations.serialize(self.activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
        }
        base_config = super().get_config()
        return {**base_config, **config}
    

class WESN(tf.keras.layers.Layer):
    def __init__(
            self,
            units: int,
            connectivity: float = 0.1,
            leaky: float = 1.0,
            spectral_radius: float = 0.9,
            use_norm2: bool = False,
            use_bias: bool = True,
            activation="tanh",
            kernel_initializer="glorot_uniform",
            recurrent_initializer="glorot_uniform",
            bias_initializer="zeros",
            win_len: int = 0,
            readout_units: int = 1,
            input_scale: float = 1.0,
            complex_input: bool = False,
            inv_regularization: int = 1,
            **kwargs):
        super().__init__(dtype=tf.complex64, **kwargs)
        self.cell = ESNCell(
                units=units,
                connectivity=connectivity,
                leaky=leaky,
                spectral_radius=spectral_radius,
                use_norm2=use_norm2,
                use_bias=use_bias,
                activation=activation,
                kernel_initializer=kernel_initializer,
                recurrent_initializer=recurrent_initializer,
                bias_initializer=bias_initializer,
            )
        self.reservoir = tf.keras.layers.RNN(self.cell, return_sequences=True, return_state=False)
        self.win_len = win_len
        self.readout_units = readout_units
        self.input_scale = input_scale
        self.complex_input = complex_input
        self.inv_regularization = inv_regularization
        # Bias is deliberately disabled so the readout is a pure linear mapping
        self.readout = tf.keras.layers.Dense(
            readout_units,
            use_bias=False,
            dtype=tf.complex64,
            kernel_initializer=complex_initializer(tf.keras.initializers.GlorotUniform()),
        )

    # The Dense read‑out is created lazily in build() once the input dimensionality is known
    def build(self, inputs_shape):
        batch, time_steps, input_dim = inputs_shape

        # total feature dim = reservoir_units + win_len * input_dim
        feature_dim = self.cell.units + input_dim * self.win_len

        # now we know the read-out’s input size, build it explicitly
        # Dense can handle higher-rank inputs, but build expects at least (None, feature_dim)
        self.readout.build(tf.TensorShape([None, feature_dim]))
        
        super().build(inputs_shape)

    def _stack_inputs(self, inputs):
        """Run the reservoir and stack optional input lags."""
        inputs = tf.cast(inputs, tf.complex64) * self.input_scale
        res_out = self.reservoir(inputs)  # [batch, timesteps, units]

        stacked_inp = [res_out]
        # Make sure win_len <= inputs.shape[1]
        if self.win_len:
            if self.win_len > inputs.shape[1]:
                raise ValueError(f"win_len ({self.win_len}) cannot be greater than input timesteps ({inputs.shape[1]})")
            stacked_inp.append(inputs)
            for k in range(1, self.win_len):
                lag_k = tf.pad(inputs[:, :-k, :], paddings=[[0, 0], [k, 0], [0, 0]])
                stacked_inp.append(lag_k)
        
        return tf.concat(stacked_inp, axis=-1)
    
    def call(self, inputs):
        # 1) run through the untrained ESN reservoir
        stacked_inp = self._stack_inputs(inputs)
        output = self.readout(stacked_inp)  # [batch, timesteps, readout_units]
        return output

    def ls_initialize(self, inputs, targets):
        """Compute a least-squares solution for the readout weights.

        The Dense readout is initialized via a pseudo-inverse solution before
        subsequent gradient-based optimisation.

        Parameters
        ----------
        inputs : tf.Tensor
            Input sequences with shape ``[batch, timesteps, features]``.
        targets : tf.Tensor
            Target sequences with shape ``[batch, timesteps, readout_units]``.
        """
        if not self.built:
            self.build(inputs.shape)

        inputs = tf.cast(inputs, tf.complex64)
        targets = tf.cast(targets, tf.complex64)
        stacked_inp = self._stack_inputs(inputs)
        shape = tf.shape(stacked_inp)
        bt = shape[0] * shape[1]
        f = shape[2]
        x = tf.reshape(stacked_inp, [bt, f])
        y = tf.reshape(targets, [bt, -1])
        y = tf.cast(y, dtype=x.dtype)

        # Solve for weights without an explicit bias term
        w = self.reg_p_inv(x) @ y  # [feature_dim, readout_units]
        self.readout.kernel.assign(w)
    
    def reg_p_inv(self, X):
        """
        Compute regularized pseudoinverse:
        (Xᴴ (X Xᴴ + reg*I)⁻¹)

        Args:
            X: [N, F] complex tensor
            reg: ridge parameter (float)

        Returns:
            [F, N] complex tensor (regularized pseudoinverse of X)
        """
        X = tf.convert_to_tensor(X)
        N = tf.shape(X)[0]

        # Identity [N, N]
        I = tf.eye(N, dtype=X.dtype)

        # X @ Xᴴ   [N, N]
        XXH = tf.matmul(X, X, adjoint_b=True)

        # (XXᴴ + reg I)⁻¹   [N, N]
        inv_term = tf.linalg.inv(XXH + tf.cast(self.inv_regularization, X.dtype) * I)

        # Xᴴ @ inv_term   [F, N]
        return tf.matmul(X, inv_term, adjoint_a=True)



