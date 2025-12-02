from tensorflow.keras.layers import Input, SimpleRNN, Dense
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from .metrics import rmse, r2

import tensorflow as tf


class MySimpleRNN(layers.Layer):
    def __init__(self, units, activation="tanh", return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        # input_shape: (batch_size, time_steps, input_dim)
        input_dim = input_shape[-1]

        self.W_xh = self.add_weight(
            name="W_xh",
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.W_hh = self.add_weight(
            name="W_hh",
            shape=(self.units, self.units),
            initializer="orthogonal",
            trainable=True,
        )

        self.b_h = self.add_weight(
            name="b_h",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):
        """
        inputs: (batch_size, time_steps, input_dim)
        """
        batch_size = tf.shape(inputs)[0]
        time_steps = inputs.shape[1]   # <-- LẤY STATIC DIM, LÀ SỐ NGUYÊN

        h_t = tf.zeros((batch_size, self.units), dtype=inputs.dtype)

        if self.return_sequences:
            outputs = []

        # Dùng range(time_steps) với time_steps là int Python, KHÔNG phải tensor
        for t in range(time_steps):
            x_t = inputs[:, t, :]  # (batch_size, input_dim)

            h_t = self.activation(
                tf.matmul(x_t, self.W_xh) + tf.matmul(h_t, self.W_hh) + self.b_h
            )

            if self.return_sequences:
                outputs.append(h_t)

        if self.return_sequences:
            # outputs: list length T, mỗi phần tử (batch_size, units)
            outputs = tf.stack(outputs, axis=1)  # (batch_size, time_steps, units)
            return outputs
        else:
            return h_t  # (batch_size, units)

    def compute_output_shape(self, input_shape):
        """
        Cho Keras biết trước shape output để nó không phải "đoán".
        """
        batch_size, time_steps, _ = input_shape
        if self.return_sequences:
            return (batch_size, time_steps, self.units)
        else:
            return (batch_size, self.units)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": tf.keras.activations.serialize(self.activation),
                "return_sequences": self.return_sequences,
            }
        )
        return config



def build_2step_model(
    lookback: int,
    num_features: int,
    learning_rate: float = 0.001,
):
    """
    Simple RNN model dự báo PM2.5 tại thời điểm t+2.
    """

    # 1. Input
    inputs = Input(shape=(lookback, num_features), name="rnn_input")

    # 2. Dùng custom MySimpleRNN thay vì keras.layers.SimpleRNN
    x = MySimpleRNN(
        units=64,
        activation="tanh",
        return_sequences=False,
        name="mysimplernn_64_tanh",
    )(inputs)

    # 3. Dense hidden
    x = layers.Dense(
        units=32,
        activation="relu",
        name="dense_32_relu",
    )(x)

    # 4. Output
    outputs = layers.Dense(
        units=1,
        activation=None,
        name="output_pm25_t_plus_2",
    )(x)

    # 5. Model
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="simple_rnn_2step_pm25",
    )

    # 6. Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[rmse, r2],
    )

    return model
