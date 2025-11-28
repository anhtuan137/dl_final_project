# PM2.5 Forecasting with Deep Learning (RNN & Seq2Seq)

Dự án này xây dựng các mô hình **deep learning** để dự báo nồng độ **PM2.5** trong không khí dựa trên dữ liệu chuỗi thời gian (AQI và khí tượng).

Gồm 2 bài toán chính:

1. **Task 1 – 2-step forecasting (Simple RNN)**  
   Dự báo giá trị `PM2.5` tại thời điểm **t+2** từ chuỗi quan sát quá khứ.

2. **Task 2 – 5-step forecasting (LSTM Seq2Seq)**  
   Dự báo giá trị `PM2.5` cho **5 bước thời gian liên tiếp**: t+1, t+2, t+3, t+4, t+5  
   sử dụng mô hình **Encoder–Decoder (sequence-to-sequence) với LSTM**.

Cả hai mô hình đều được đánh giá bằng các metric:

- **RMSE** – Root Mean Squared Error  
- **MAE** – Mean Absolute Error  
- **R²** – Coefficient of Determination  
- **NSE** – Nash–Sutcliffe Efficiency

---

## 1. Mục tiêu & Ý tưởng chính

### 1.1. Biến dự báo: `pred_column = PM2.5`

Ta chọn một thuộc tính duy nhất làm biến cần dự báo:

```python
PRED_COLUMN = "PM2.5"
```

Các đặc trưng đầu vào (feature) gồm toàn bộ các cột numeric khác, ví dụ:

- PM2.5, PM10, SO2, NO2, CO, O3  
- TEM, PRES, DEMP, RAIN, WSPM  

(tùy thuộc dữ liệu thực tế trong các file CSV).

### 1.2. Task 1: Forecast t+2 (Simple RNN)

- **Input**: chuỗi độ dài `lookback` (ví dụ 12 bước), mỗi bước là vector nhiều feature.  
  Shape: `(batch_size, lookback, num_features)`
- **Output**: 1 giá trị **PM2.5 tại t+2**.  
  Shape: `(batch_size, 1)`

Mô hình: **SimpleRNN → Dense → Dense(1)**, compile với:

- `loss="mse"`
- `metrics=[rmse, r2]` (các metric custom định nghĩa trong `src/metrics.py`)

### 1.3. Task 2: Forecast t+1..t+5 (LSTM Seq2Seq)

- **Input**: chuỗi độ dài `lookback` giống Task 1.  
- **Output**: chuỗi dài `horizon = 5`, mỗi bước 1 giá trị PM2.5.  
  Shape: `(batch_size, horizon, 1)`

Mô hình: **LSTM Encoder–Decoder**:

- **Encoder**: LSTM đọc chuỗi input, trả về trạng thái ẩn (h, c).
- **Decoder**: `RepeatVector(horizon)` + LSTM (với initial_state) + `TimeDistributed(Dense(1))`.

---

## 2. Cấu trúc dự án

```text
final_project/
├── data/
│   ├── xxx_1.csv
│   ├── xxx_2.csv
│   └── ...
├── results/
│   ├── checkpoints/
│   │   ├── pm25_2step.weights.h5
│   │   └── pm25_5step_seq2seq.weights.h5
│   ├── figures_2step/
│   │   ├── loss_2step.png
│   │   ├── rmse_2step.png
│   │   ├── r2_2step.png
│   │   ├── pm25_2step_compare_first200.png
│   │   └── pm25_2step_test_metrics.png
│   ├── figures_5step/
│   │   ├── loss_5step_seq2seq.png
│   │   ├── rmse_5step_seq2seq.png
│   │   ├── r2_5step_seq2seq.png
│   │   ├── pm25_5step_seq2seq_tplus1_compare.png
│   │   └── pm25_5step_seq2seq_test_metrics.png
│   └── logs/
│       ├── metrics_2step.txt
│       └── metrics_5step_seq2seq.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model_builder_2step.py
│   ├── model_builder_seq2seq.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualization.py
│   └── metrics.py
├── main_2step.py
├── main_5step_seq2seq.py
├── requirements.txt
└── README.md
```

---

## 3. Chuẩn bị môi trường

### 3.1. Yêu cầu chung

- Python **3.11**  
- `pip` mới (>= 22)  
- Khuyến nghị: sử dụng **virtual environment (.venv)** riêng trong thư mục dự án.

### 3.2. Tạo môi trường ảo (venv)

Trong thư mục `final_project/`:

```bash
# Tạo venv (macOS / Linux)
python3 -m venv .venv

# Kích hoạt venv
source .venv/bin/activate
```

Trên Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scriptsctivate
```

### 3.3. Cài dependencies

Trong môi trường ảo:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Ví dụ nội dung `requirements.txt`:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
optuna
```

> ⚠️ Trên macOS M1/M2 có thể cần `tensorflow-macos` hoặc cấu hình riêng cho TensorFlow tuỳ môi trường.

---

## 4. Dữ liệu đầu vào

### 4.1. Thư mục `data/`

- Chứa nhiều file CSV (dữ liệu AQI, khí tượng) có cấu trúc tương tự.
- Hàm `src.data_loader.load_dataset` sẽ:
  - Đọc **tất cả** file `.csv` trong thư mục `data/`.
  - Gộp lại thành một DataFrame bằng `pd.concat`.
  - Loại bỏ các cột không dùng:
    - `['No', 'year', 'month', 'day', 'hour', 'wd', 'station']`
  - Xử lý missing value bằng **backfill** (`fillna_back`).

### 4.2. Xử lý thiếu, ngoại lai, mất cân bằng, tăng cường dữ liệu

Trong `src/preprocessing.py`:

- **Thiếu dữ liệu**:
  - `fillna_back(df)` – dùng giá trị quan sát phía sau (backfill) để lấp missing.
  - (Option khác: `fillna_mean(df)` – điền bằng mean theo cột.)

- **Ngoại lai (outlier)**:
  - `clip_outliers_iqr(df, factor=1.5)` – clip các giá trị nằm ngoài  
    `[Q1 - factor * IQR, Q3 + factor * IQR]` cho các cột numeric.

- **Tăng cường dữ liệu (data augmentation)**:
  - `augment_sequences_gaussian(X, y, n_aug, noise_std)`  
    → thêm nhiễu Gaussian nhỏ lên `X` (sau khi scale), giúp mô hình robust hơn.

- **Mất cân bằng dữ liệu (nhấn mạnh PM2.5 cao)**:
  - `oversample_high_values(X, y, high_quantile, n_repeat)`  
    → lặp lại các mẫu có giá trị target thuộc top `high_quantile` (ví dụ 0.8 → top 20% lớn nhất).

---

## 5. Chuẩn bị dữ liệu cho mô hình

### 5.1. Task 1 – 2-step forecasting (t+2)

```python
from src.preprocessing import prepare_data_2step

train_X, train_y, val_X, val_y, test_X, test_y, scaler = prepare_data_2step(
    dataset,
    pred_column="PM2.5",
    lookback=12,
    train_ratio=0.7,
    valid_ratio=0.9,
    # Các tuỳ chọn dưới đây có thể bật/tắt:
    # clip_outliers=True,
    # augment_train=True,
    # aug_n=2,
    # aug_noise_std=0.01,
    # balance_high=True,
    # high_quantile=0.8,
    # high_repeat=2,
)
```

Hàm sẽ:

- Xây **cửa sổ thời gian** độ dài `lookback`.
- Label là PM2.5 tại **t+2**.
- Scale toàn bộ feature bằng `MinMaxScaler(0,1)`.
- Chia dữ liệu thành train / val / test theo tỉ lệ chỉ định.

Kết quả:

- `train_X.shape = (N_train, lookback, num_features)`
- `train_y.shape = (N_train, 1)`

### 5.2. Task 2 – 5-step Seq2Seq (t+1..t+5)

```python
from src.preprocessing import prepare_data_5step_seq2seq

train_X, train_y, val_X, val_y, test_X, test_y, scaler = prepare_data_5step_seq2seq(
    dataset,
    pred_column="PM2.5",
    lookback=12,
    horizon=5,
    train_ratio=0.7,
    valid_ratio=0.9,
    # clip_outliers=True,
    # augment_train=True,
    # aug_n=2,
    # aug_noise_std=0.01,
    # balance_high=True,
    # high_quantile=0.8,
    # high_repeat=2,
)
```

Hàm sẽ:

- Tạo dữ liệu đầu vào (lookback bước) với toàn bộ feature.
- Output là chuỗi PM2.5 `[t+1, t+2, t+3, t+4, t+5]`.

Kết quả:

- `train_X.shape = (N_train, lookback, num_features)`
- `train_y.shape = (N_train, horizon, 1)`

---

## 6. Mô hình

### 6.1. Task 1 – Simple RNN dự báo t+2

File: `src/model_builder_2step.py`

Mô hình Simple RNN được code **cụ thể từng layer**, ví dụ:

```python
from tensorflow.keras.layers import Input, SimpleRNN, Dense
from tensorflow.keras.models import Model
from tensorflow import keras

from .metrics import rmse, r2

def build_2step_model(lookback, num_features, learning_rate=0.001):
    inputs = Input(shape=(lookback, num_features), name="rnn_input")

    x = SimpleRNN(
        units=64,
        activation="tanh",
        return_sequences=False,
        name="simplernn_64_tanh",
    )(inputs)

    x = Dense(
        units=32,
        activation="relu",
        name="dense_32_relu",
    )(x)

    outputs = Dense(
        units=1,
        activation=None,
        name="output_pm25_t_plus_2",
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="simple_rnn_2step_pm25")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[rmse, r2],
    )
    return model
```

### 6.2. Task 2 – LSTM Seq2Seq dự báo 5 bước

File: `src/model_builder_seq2seq.py`

Mô hình Encoder–Decoder với LSTM:

- Encoder LSTM đọc `(lookback, num_features)` → trả về `(state_h, state_c)`.
- Decoder:
  - `RepeatVector(horizon)` để lặp trạng thái ẩn theo chiều thời gian.
  - LSTM với `return_sequences=True`, khởi tạo `initial_state=[state_h, state_c]`.
  - `TimeDistributed(Dense(1))` để sinh ra chuỗi dự báo PM2.5.

Compile với:

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss="mse",
    metrics=[rmse, r2],
)
```

---

## 7. Huấn luyện

File: `src/train.py`

### 7.1. Callbacks

- **EarlyStopping**:
  - `monitor="val_loss"`
  - `patience` (ví dụ 10 hoặc 20)
  - `restore_best_weights=True`
- **ModelCheckpoint**:
  - Lưu weights tốt nhất vào `results/checkpoints/*.weights.h5`.

### 7.2. Vòng lặp huấn luyện

```python
history = model.fit(
    train_X,
    train_y,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(val_X, val_y),
    callbacks=callbacks,
)
```

---

## 8. Đánh giá (Evaluation)

File: `src/evaluate.py`

### 8.1. Metric tính trên **dữ liệu thật (unscaled)**

Các bước:

1. Model dự đoán trên test set (dữ liệu đã scale).
2. Dùng min/max của `PRED_COLUMN` để inverse-scale.
3. Tính các metric:
   - RMSE  
   - MAE  
   - R² (từ `sklearn.metrics.r2_score`)  
   - NSE (Nash–Sutcliffe Efficiency) – custom trong file.

### 8.2. `evaluate_2step(...)`

- Input: `model`, `test_X`, `test_y` (scaled), `dataset`, `pred_column`.
- Output:
  - `y_true`, `y_pred` (PM2.5 thật, shape `(N, 1)`).
  - `(rmse, mae, r2, nse)`.

Log được ghi vào: `results/logs/metrics_2step.txt`.

### 8.3. `evaluate_5step_seq2seq(...)`

- Input: `model`, `test_X`, `test_y` (shape `(N, horizon, 1)`), `dataset`, `pred_column`.
- Output:
  - `y_true`, `y_pred` (shape `(N, horizon)` sau squeeze).
  - `(rmse, mae, r2, nse)` tính trên toàn bộ chuỗi (flatten N*horizon).

Log được ghi vào: `results/logs/metrics_5step_seq2seq.txt`.

---

## 9. Visualization

File: `src/visualization.py`

Các hàm chính:

- `visualize_loss(history, title, save_path)`  
  → loss & val_loss theo epoch.

- `visualize_rmse(history, title, save_path)`  
  → RMSE & val_RMSE theo epoch (metrics từ `model.compile`).

- `visualize_r2(history, title, save_path)`  
  → R² & val_R² theo epoch.

- `visualize_pm25_comparison(y_true, y_pred, n_points, title, save_path)`  
  → vẽ đường **so sánh PM2.5 thực vs dự đoán** trên test (ví dụ 200 mẫu đầu).

- `visualize_test_metrics(rmse, mae, r2, nse, title, save_path)`  
  → biểu đồ bar 4 metric cuối cùng trên test.

- `compare_visual(data1, data2, columns, ...)`  
  → so sánh nhiều biến cùng lúc (nếu cần).

---

## 10. Cách chạy

### 10.1. Chạy Task 1 – Simple RNN 2-step

Trong thư mục `final_project/`:

```bash
source .venv/bin/activate
python main_2step.py
```

`main_2step.py` sẽ:

1. Load dữ liệu từ `data/`.
2. Gọi `prepare_data_2step` để tạo train/val/test.
3. Build model Simple RNN với `build_2step_model`.
4. Train mô hình, lưu checkpoint tốt nhất.
5. Vẽ:
   - `results/figures_2step/loss_2step.png`
   - `results/figures_2step/rmse_2step.png`
   - `results/figures_2step/r2_2step.png`
6. Evaluate trên test:
   - Log metrics trong `results/logs/metrics_2step.txt`
   - Vẽ:
     - `results/figures_2step/pm25_2step_compare_first200.png`
     - `results/figures_2step/pm25_2step_test_metrics.png`

### 10.2. Chạy Task 2 – LSTM Seq2Seq 5-step

```bash
source .venv/bin/activate
python main_5step_seq2seq.py
```

`main_5step_seq2seq.py` sẽ:

1. Load dữ liệu từ `data/`.
2. Gọi `prepare_data_5step_seq2seq` để tạo train/val/test.
3. Build Seq2Seq LSTM bằng `build_seq2seq_model`.
4. Train mô hình, lưu checkpoint.
5. Vẽ:
   - `results/figures_5step/loss_5step_seq2seq.png`
   - `results/figures_5step/rmse_5step_seq2seq.png`
   - `results/figures_5step/r2_5step_seq2seq.png`
6. Evaluate trên test:
   - Log metrics trong `results/logs/metrics_5step_seq2seq.txt`
   - Vẽ:
     - `results/figures_5step/pm25_5step_seq2seq_test_metrics.png`
     - `results/figures_5step/pm25_5step_seq2seq_tplus1_compare.png` (có thể chọn bước t+2, t+3, t+4, t+5 tùy cấu hình).

---

## 11. Cách đọc và diễn giải kết quả

### 11.1. Metric

- **RMSE**: càng nhỏ càng tốt, nhạy với outlier.  
- **MAE**: dễ diễn giải (đơn vị giống PM2.5).  
- **R²**:
  - Gần 1: mô hình giải thích tốt phương sai.
  - Gần 0: không tốt hơn đoán theo trung bình.
  - < 0: tệ hơn cả đoán theo trung bình.
- **NSE**:
  - 1: hoàn hảo.
  - 0: tương đương dự đoán bằng trung bình.
  - < 0: tệ hơn trung bình.

### 11.2. Hình vẽ

- **Loss / RMSE / R² theo epoch**:
  - Kiểm tra overfitting/underfitting.
  - Kiểm tra mô hình có học thêm được sau nhiều epoch không.

- **Actual vs Predicted (PM2.5)**:
  - Xem mô hình có bám sát đường biến động thực không.
  - Quan sát các giai đoạn PM2.5 rất cao hoặc rất thấp.

- **Bar chart test metrics**:
  - Dùng cho báo cáo, so sánh giữa mô hình 2-step và 5-step.

---

## 12. Tuỳ chỉnh & Mở rộng

- Có thể đổi `PRED_COLUMN` sang biến khác (ví dụ `PM10`) nếu cột có trong data.
- Điều chỉnh:
  - `lookback`, `horizon`
  - Các tham số tăng cường dữ liệu (`aug_n`, `aug_noise_std`)
  - Oversampling (`high_quantile`, `high_repeat`)
- Thử mô hình khác:
  - LSTM/GRU cho Task 1.
  - Attention trên encoder/decoder.
- Thêm phân tích:
  - Metric theo từng khoảng PM2.5 (thấp – trung bình – cao).
  - So sánh mô hình chỉ dùng PM2.5 vs dùng thêm các feature khí tượng.

---

## 13. Hướng dẫn nhanh cho người mới mở project

1. Clone / copy thư mục `final_project`.
2. Đặt các file CSV vào thư mục `data/`.
3. Tạo venv + cài `requirements.txt`.
4. Chạy:
   - `python main_2step.py`
   - `python main_5step_seq2seq.py`
5. Xem kết quả trong thư mục `results/`:
   - `logs/` – metric số.
   - `figures_2step/` và `figures_5step/` – hình vẽ phục vụ báo cáo.
