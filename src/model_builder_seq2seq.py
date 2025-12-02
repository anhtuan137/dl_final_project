import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input,
    RNN,
    RepeatVector,
    TimeDistributed,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow import keras

from .metrics import rmse, r2


class MyLSTMCell(layers.Layer):
    """
    LSTM cell tự cài, tương đương về mặt toán học với keras.layers.LSTMCell (đơn giản hóa).

    - units: số chiều hidden state h_t và cell state c_t.
    - activation: dùng cho candidate cell state (thường là 'tanh').
    - recurrent_activation: dùng cho các gate i,f,o (thường là 'sigmoid').

    call(inputs, states):
        inputs: (batch_size, input_dim)
        states: [h_{t-1}, c_{t-1}]
        return: (h_t, [h_t, c_t])
    """

    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)

    @property
    def state_size(self):
        # LSTM có 2 state: h và c, mỗi cái size = units
        return [self.units, self.units]

    @property
    def output_size(self):
        # output mỗi time step là h_t
        return self.units

    def build(self, input_shape):
        """
        input_shape: (batch_size, input_dim)
        """
        input_dim = input_shape[-1]

        # Kernel cho input: x_t @ W (input_dim, 4*units)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, 4 * self.units),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Recurrent kernel cho hidden: h_{t-1} @ U (units, 4*units)
        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=(self.units, 4 * self.units),
            initializer="orthogonal",
            trainable=True,
        )

        # Bias: (4*units,)
        self.bias = self.add_weight(
            name="bias",
            shape=(4 * self.units,),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, states):
        """
        inputs: (batch_size, input_dim)
        states: [h_{t-1}, c_{t-1}]
        """
        h_tm1, c_tm1 = states  # previous hidden state, previous cell state

        # z = x_t @ W + h_{t-1} @ U + b  -> shape: (batch_size, 4*units)
        z = tf.matmul(inputs, self.kernel) + tf.matmul(h_tm1, self.recurrent_kernel) + self.bias

        # Chia z thành 4 phần cho 4 "gate": i, f, g, o
        z0, z1, z2, z3 = tf.split(z, num_or_size_splits=4, axis=1)

        i = self.recurrent_activation(z0)      # input gate
        f = self.recurrent_activation(z1)      # forget gate
        g = self.activation(z2)                # candidate cell
        o = self.recurrent_activation(z3)      # output gate

        # Cập nhật cell state và hidden state
        c_t = f * c_tm1 + i * g
        h_t = o * self.activation(c_t)

        return h_t, [h_t, c_t]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": tf.keras.activations.serialize(self.activation),
                "recurrent_activation": tf.keras.activations.serialize(
                    self.recurrent_activation
                ),
            }
        )
        return config

def build_seq2seq_model(
    lookback: int,
    num_features: int,
    horizon: int = 5,
    learning_rate: float = 0.001,
):
    """
    Encoder–Decoder (Seq2Seq) với LSTM (custom cell) để dự báo pred_column (PM2.5)
    cho 5 bước liên tiếp trong tương lai.

    - Input  : (batch_size, lookback, num_features)
    - Output : (batch_size, horizon, 1)
    """

    # ================== 1. ENCODER ==================
    encoder_inputs = Input(
        shape=(lookback, num_features),
        name="encoder_input",
    )

    # Encoder: RNN với MyLSTMCell
    # return_sequences=False: chỉ lấy output cuối cùng (h_T)
    # return_state=True: trả thêm h_T, c_T
    encoder_lstm = RNN(
        MyLSTMCell(
            units=64,
            activation="tanh",
            recurrent_activation="sigmoid",
            name="encoder_lstm_cell_64",
        ),
        return_sequences=False,
        return_state=True,
        name="encoder_lstm_64",
    )

    encoder_output, state_h, state_c = encoder_lstm(encoder_inputs)
    # encoder_output == state_h trong cấu hình này, nhưng ta dùng state_h, state_c cho decoder.

    # ================== 2. PREPARE DECODER INPUT ==================
    decoder_inputs = RepeatVector(horizon, name="repeat_horizon")(state_h)

    # ================== 3. DECODER LSTM ==================
    # Decoder: cũng dùng MyLSTMCell với cùng units để dùng chung state
    decoder_lstm = RNN(
        MyLSTMCell(
            units=64,
            activation="tanh",
            recurrent_activation="sigmoid",
            name="decoder_lstm_cell_64",
        ),
        return_sequences=True,
        return_state=False,
        name="decoder_lstm_64",
    )

    decoder_outputs = decoder_lstm(
        decoder_inputs,
        initial_state=[state_h, state_c],
    )
    # decoder_outputs shape: (batch_size, horizon, 64)

    # ================== 4. TIME-DISTRIBUTED DENSE OUTPUT ==================
    decoder_dense = TimeDistributed(
        Dense(
            units=1,
            activation=None,  # linear cho regression
            name="pm25_output_step",
        ),
        name="timedistributed_pm25",
    )

    outputs = decoder_dense(decoder_outputs)
    # outputs shape: (batch_size, horizon, 1)

    # ================== 5. DEFINE MODEL ==================
    model = Model(
        inputs=encoder_inputs,
        outputs=outputs,
        name="seq2seq_lstm_pm25_5step_custom",
    )

    # ================== 6. COMPILE MODEL ==================
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[rmse, r2],
    )

    return model
