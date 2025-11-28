from tensorflow.keras.layers import Input, SimpleRNN, Dense
from tensorflow.keras.models import Model
from tensorflow import keras

from .metrics import rmse, r2


def build_2step_model(
    lookback: int,
    num_features: int,
    learning_rate: float = 0.001,
):
    """
    Simple RNN model dự báo pred_column (PM2.5) tại thời điểm t+2.

    - Input  : chuỗi quá khứ có độ dài = lookback, với num_features đặc trưng mỗi bước
               shape = (batch_size, lookback, num_features)
    - Output : 1 giá trị scalar cho PM2.5 tại t+2
               shape = (batch_size, 1)
    """

    # ====== 1. Định nghĩa input ======
    # Mỗi sample là một chuỗi: [x_{t-lookback+1}, ..., x_t]
    inputs = Input(shape=(lookback, num_features), name="rnn_input")

    # ====== 2. SimpleRNN layer ======
    # units=64: số lượng "hidden units" trong trạng thái ẩn
    # activation='tanh': chuẩn cho RNN cổ điển, giúp tránh exploding gradient tốt hơn so với 'relu'
    # return_sequences=False: chỉ lấy output cuối cùng tại thời điểm t (đại diện cho cả chuỗi)
    x = SimpleRNN(
        units=64,
        activation="tanh",
        return_sequences=False,
        name="simplernn_64_tanh",
    )(inputs)

    # ====== 3. Fully-connected hidden layer ======
    # Dense(32) để học thêm biến đổi phi tuyến từ hidden state RNN
    x = Dense(
        units=32,
        activation="relu",
        name="dense_32_relu",
    )(x)

    # ====== 4. Output layer ======
    # Dự báo 1 giá trị: PM2.5 tại t+2 (được scale, sẽ inverse ở evaluate)
    outputs = Dense(
        units=1,
        activation=None,   # linear
        name="output_pm25_t_plus_2",
    )(x)

    # ====== 5. Kết nối thành Model ======
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="simple_rnn_2step_pm25",
    )

    # ====== 6. Compile model ======
    # - loss='mse' cho bài toán regression
    # - metrics=[rmse, r2] để vẽ RMSE/R² theo epoch trong history
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[rmse, r2],
    )

    return model
