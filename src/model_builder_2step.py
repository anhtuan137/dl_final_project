import tensorflow as tf
from tensorflow.keras import layers, Model, Input

from .metrics import rmse, r2


class MySimpleRNN(layers.Layer):
    """
    Custom SimpleRNN:
    - Tự cài phương trình RNN cơ bản:
        h_t = activation(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
    - Có thêm:
        + recurrent_dropout: dropout trên nhánh h_{t-1} -> h_t (chỉ dùng khi training=True)
    """

    def __init__(
        self,
        units,
        activation="tanh",
        return_sequences=False,
        recurrent_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.return_sequences = return_sequences
        self.recurrent_dropout = float(recurrent_dropout)

    def build(self, input_shape):
        # input_shape: (batch_size, time_steps, input_dim)
        input_dim = input_shape[-1]

        # Trọng số input -> hidden
        self.W_xh = self.add_weight(
            name="W_xh",
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Trọng số hidden -> hidden (recurrent)
        self.W_hh = self.add_weight(
            name="W_hh",
            shape=(self.units, self.units),
            initializer="orthogonal",
            trainable=True,
        )

        # Bias
        self.b_h = self.add_weight(
            name="b_h",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        inputs:  (batch_size, time_steps, input_dim)
        training: bool hoặc None. Nếu True -> bật recurrent dropout.
        """
        batch_size = tf.shape(inputs)[0]
        # static dim cho số bước thời gian
        time_steps = inputs.shape[1]

        # hidden state ban đầu = 0
        h_t = tf.zeros((batch_size, self.units), dtype=inputs.dtype)

        # chuẩn bị list output nếu cần trả cả sequence
        if self.return_sequences:
            outputs = []

        # Tạo mask cho recurrent dropout (ẩn -> ẩn), giữ cố định cho toàn bộ chuỗi
        recurrent_mask = None
        if training and 0.0 < self.recurrent_dropout < 1.0:
            # mask shape = (batch_size, units), các phần tử ~1/(1-rate) hoặc 0
            recurrent_mask = tf.nn.dropout(
                tf.ones_like(h_t),
                rate=self.recurrent_dropout,
            )

        # Dùng range(time_steps) với time_steps là int Python
        for t in range(time_steps):
            x_t = inputs[:, t, :]  # (batch_size, input_dim)

            h_prev = h_t
            if recurrent_mask is not None:
                # recurrent dropout: tắt ngẫu nhiên một số chiều của h_{t-1}
                h_prev = h_prev * recurrent_mask

            # h_t = activation(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
            h_t = self.activation(
                tf.matmul(x_t, self.W_xh) + tf.matmul(h_prev, self.W_hh) + self.b_h
            )

            if self.return_sequences:
                outputs.append(h_t)

        if self.return_sequences:
            # outputs: list length T, mỗi phần tử (batch_size, units)
            outputs = tf.stack(outputs, axis=1)  # (batch_size, time_steps, units)
            return outputs
        else:
            # chỉ trả hidden state cuối cùng
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
                "recurrent_dropout": self.recurrent_dropout,
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

    - Có:
        + MySimpleRNN (với recurrent_dropout)
        + Layer Normalization sau RNN
        + 1 residual block 64 chiều (Dense -> Dense + residual)
        + Dense cuối cùng để ra 1 giá trị dự báo PM2.5(t+2)
    """

    # 1. Input
    inputs = Input(shape=(lookback, num_features), name="rnn_input")

    # 2. Custom MySimpleRNN với recurrent_dropout
    x = MySimpleRNN(
        units=64,
        activation="tanh",
        return_sequences=False,
        recurrent_dropout=0.2,   # recurrent dropout 20% trên h_{t-1}
        name="mysimplernn_64_tanh",
    )(inputs)

    # 3. Layer Normalization trên output của RNN
    x = layers.LayerNormalization(name="rnn_layer_norm")(x)

    # 4. Residual block 64 chiều
    #    - Branch chính: Dense(64) -> ReLU -> Dense(64)
    #    - Residual: cộng với đầu vào của block (x_residual)
    x_residual = x  # lưu lại để cộng residual
    x = layers.Dense(
        units=64,
        activation="relu",
        name="res_block_dense1_64_relu",
    )(x)
    x = layers.Dense(
        units=64,
        activation="relu",
        name="res_block_dense2_64_relu",
    )(x)
    x = layers.Add(name="residual_add_64")([x, x_residual])

    # 5. Dense ẩn cuối 32 chiều
    x = layers.Dense(
        units=32,
        activation="relu",
        name="dense_32_relu",
    )(x)

    # 6. Output: 1 giá trị PM2.5 tại t+2 (trên không gian đã scale)
    outputs = layers.Dense(
        units=1,
        activation=None,
        name="output_pm25_t_plus_2",
    )(x)

    # 7. Model
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="simple_rnn_2step_pm25_with_norm_residual_dropout",
    )

    # 8. Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[rmse, r2],
    )

    return model
