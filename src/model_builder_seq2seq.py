from tensorflow.keras.layers import (
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow import keras

from .metrics import rmse, r2


def build_seq2seq_model(
    lookback: int,
    num_features: int,
    horizon: int = 5,
    learning_rate: float = 0.001,
):
    """
    Encoder–Decoder (Seq2Seq) với LSTM để dự báo pred_column (PM2.5)
    cho 5 bước liên tiếp trong tương lai.

    - Input  : chuỗi quá khứ length = lookback, mỗi bước có num_features đặc trưng
               shape = (batch_size, lookback, num_features)
    - Output : chuỗi tương lai length = horizon (mặc định 5),
               mỗi bước là 1 giá trị PM2.5 đã scale
               shape = (batch_size, horizon, 1)
    """

    # ================== 1. ENCODER ==================
    # Mỗi mẫu là một chuỗi: [x_{t-lookback+1}, ..., x_t]
    encoder_inputs = Input(
        shape=(lookback, num_features),
        name="encoder_input",
    )

    # LSTM encoder:
    # - units = 64: kích thước trạng thái ẩn (hidden state) và cell state
    # - return_sequences = False: chỉ lấy output tại thời điểm cuối cùng
    # - return_state = True: lấy cả (output_last, h, c)
    encoder_lstm = LSTM(
        units=64,
        activation="tanh",
        recurrent_activation="sigmoid",
        return_sequences=False,
        return_state=True,
        name="encoder_lstm_64",
    )

    encoder_output, state_h, state_c = encoder_lstm(encoder_inputs)
    # encoder_output thường trùng với state_h khi return_sequences=False,
    # nhưng ta dùng state_h, state_c làm "context" cho decoder.

    # ================== 2. PREPARE DECODER INPUT ==================
    # RepeatVector(horizon): lặp lại vector trạng thái ẩn H lần
    # để tạo input dạng chuỗi cho decoder.
    decoder_inputs = RepeatVector(horizon, name="repeat_horizon")(state_h)

    # ================== 3. DECODER LSTM ==================
    # Decoder LSTM:
    # - units = 64: cùng kích thước với encoder để dùng chung state
    # - return_sequences = True: cần output cho mỗi time step trong chuỗi tương lai
    decoder_lstm = LSTM(
        units=64,
        activation="tanh",
        recurrent_activation="sigmoid",
        return_sequences=True,
        return_state=False,
        name="decoder_lstm_64",
    )

    # Truyền initial_state = [state_h, state_c] từ encoder sang decoder:
    decoder_outputs = decoder_lstm(
        decoder_inputs,
        initial_state=[state_h, state_c],
    )
    # decoder_outputs shape: (batch_size, horizon, 64)

    # ================== 4. TIME-DISTRIBUTED DENSE OUTPUT ==================
    # Với mỗi bước trong horizon, ta dùng 1 Dense(1) để dự báo PM2.5
    # TimeDistributed giúp áp dụng cùng một Dense(1) cho mọi time step.
    decoder_dense = TimeDistributed(
        Dense(
            units=1,
            activation=None,         # linear cho regression
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
        name="seq2seq_lstm_pm25_5step",
    )

    # ================== 6. COMPILE MODEL ==================
    # - loss = 'mse' cho bài toán hồi quy
    # - metrics = [rmse, r2] (trên giá trị đã scale)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[rmse, r2],
    )

    return model
