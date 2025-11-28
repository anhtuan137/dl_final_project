import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ======================== MISSING VALUES ========================

def fillna_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Điền giá trị thiếu bằng mean theo cột.
    (Ít dùng cho chuỗi thời gian, nhưng vẫn giữ để dùng khi cần.)
    """
    for col in list(df.columns[df.isnull().sum() > 0]):
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
    return df


def fillna_back(df: pd.DataFrame) -> pd.DataFrame:
    """
    Điền giá trị thiếu bằng backfill (giá trị phía sau).
    Thường dùng cho chuỗi thời gian AQI.
    """
    df = df.fillna(axis=0, method="bfill", limit=None)
    return df


# ======================== OUTLIER HANDLING ========================

def clip_outliers_iqr(
    df: pd.DataFrame,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Xử lý ngoại lai bằng cách CLIP (Winsorize) theo IQR:
    - Tính Q1, Q3 cho từng cột numeric
    - Giới hạn giá trị trong [Q1 - factor*IQR, Q3 + factor*IQR]

    Khuyến nghị: gọi trên toàn bộ dataset TRƯỚC khi scale.
    """
    df_clipped = df.copy()
    numeric_cols = df_clipped.select_dtypes(include=[np.number]).columns

    q1 = df_clipped[numeric_cols].quantile(0.25)
    q3 = df_clipped[numeric_cols].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    for col in numeric_cols:
        df_clipped[col] = df_clipped[col].clip(lower[col], upper[col])

    return df_clipped


# ======================== DATA AUGMENTATION ========================

def augment_sequences_gaussian(
    X: np.ndarray,
    y: np.ndarray,
    n_aug: int = 1,
    noise_std: float = 0.01,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Data augmentation cho chuỗi thời gian:
    - Thêm nhiễu Gaussian nhỏ lên X (đã scale), Y giữ nguyên.
    - Dùng cho tập TRAIN để mô hình robust hơn.

    Tham số:
        X: (n_samples, lookback, num_features)
        y: (n_samples, 1)        hoặc (n_samples, horizon, 1)
        n_aug: số lần nhân bản dữ liệu (mỗi lần thêm nhiễu khác nhau)
        noise_std: độ lệch chuẩn của noise (trên thang [0,1] sau scale)
    """
    if n_aug <= 0 or noise_std <= 0:
        return X, y

    if random_state is not None:
        np.random.seed(random_state)

    X_list = [X]
    y_list = [y]

    for _ in range(n_aug):
        noise = np.random.normal(loc=0.0, scale=noise_std, size=X.shape)
        X_noisy = X + noise
        # Đảm bảo sau khi thêm noise vẫn trong [0,1]
        X_noisy = np.clip(X_noisy, 0.0, 1.0)

        X_list.append(X_noisy)
        y_list.append(y)

    X_aug = np.concatenate(X_list, axis=0)
    y_aug = np.concatenate(y_list, axis=0)
    return X_aug, y_aug


def oversample_high_values(
    X: np.ndarray,
    y: np.ndarray,
    high_quantile: float = 0.8,
    n_repeat: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Xử lý "mất cân bằng" theo nghĩa:
    - Tăng trọng cho các mẫu có giá trị target cao (ví dụ PM2.5 cao).

    Cách làm:
        - Tìm ngưỡng quantile (vd 0.8) của y (sau scale).
        - Chọn các mẫu có y >= threshold.
        - Lặp lại (n_repeat lần) các mẫu đó trong train set.

    Thích hợp cho:
        - y shape = (n_samples, 1) cho 2-step
        - hoặc y shape = (n_samples, horizon, 1), khi đó dùng giá trị trung bình mỗi mẫu.
    """
    if n_repeat <= 0:
        return X, y

    # Flatten để tính quantile
    if y.ndim == 3:
        # (n_samples, horizon, 1) -> tính trên mean theo horizon
        y_scalar = y.mean(axis=1)  # (n_samples, 1)
    else:
        y_scalar = y

    y_scalar_flat = y_scalar.reshape(-1)
    threshold = np.quantile(y_scalar_flat, high_quantile)

    mask = y_scalar_flat >= threshold
    if not np.any(mask):
        # Không có mẫu nào "cao" hơn ngưỡng
        return X, y

    X_high = X[mask]
    y_high = y[mask]

    X_list = [X]
    y_list = [y]

    for _ in range(n_repeat):
        X_list.append(X_high)
        y_list.append(y_high)

    X_bal = np.concatenate(X_list, axis=0)
    y_bal = np.concatenate(y_list, axis=0)

    return X_bal, y_bal


# ======================== 2-STEP FORECASTING ========================

def prepare_data_2step(
    dataset: pd.DataFrame,
    pred_column: str,
    lookback: int = 12,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.9,
    clip_outliers: bool = False,
    outlier_factor: float = 1.5,
    augment_train: bool = False,
    aug_n: int = 1,
    aug_noise_std: float = 0.01,
    balance_high: bool = False,
    high_quantile: float = 0.8,
    high_repeat: int = 2,
):
    """
    Chuẩn bị dữ liệu cho bài toán (1):
    - Dự báo giá trị pred_column tại thời điểm t+2 (two time steps ahead).
    - Input: window dài 'lookback' bước, với tất cả features.
    - Output: 1 giá trị scalar (pred_column tại t+2).

    Thêm:
    - clip_outliers: nếu True, dùng IQR clipping trước khi scale.
    - augment_train: nếu True, thêm nhiễu Gaussian để làm giàu dữ liệu train.
    - balance_high: nếu True, oversample các mẫu có y cao (PM2.5 lớn).
    """

    df_proc = dataset.copy()

    # 1) Xử lý outliers (tuỳ chọn)
    if clip_outliers:
        df_proc = clip_outliers_iqr(df_proc, factor=outlier_factor)

    # 2) Scale toàn bộ feature
    values = df_proc.values.astype("float32")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    pred_idx = df_proc.columns.get_loc(pred_column)

    X_list, y_list = [], []
    n = scaled.shape[0]

    if n <= lookback + 2:
        raise ValueError("Not enough data for 2-step forecasting.")

    # t = chỉ số cuối của chuỗi input
    # ta cần t+2 còn nằm trong dữ liệu => t <= n-3
    for t in range(lookback - 1, n - 2):
        start = t - lookback + 1
        end = t + 1  # không inclusive
        X_seq = scaled[start:end, :]              # (lookback, num_features)
        y_scalar = scaled[t + 2, pred_idx]        # giá trị tại t+2

        X_list.append(X_seq)
        y_list.append(y_scalar)

    X = np.array(X_list)                         # (samples, lookback, num_features)
    y = np.array(y_list).reshape(-1, 1)         # (samples, 1)

    # 3) Chia train/val/test
    n_samples = X.shape[0]
    n_train = int(n_samples * train_ratio)
    n_valid = int(n_samples * valid_ratio)

    train_X, val_X, test_X = X[:n_train], X[n_train:n_valid], X[n_valid:]
    train_y, val_y, test_y = y[:n_train], y[n_train:n_valid], y[n_valid:]

    # 4) Data augmentation cho TRAIN (tuỳ chọn)
    if balance_high:
        train_X, train_y = oversample_high_values(
            train_X,
            train_y,
            high_quantile=high_quantile,
            n_repeat=high_repeat,
        )

    if augment_train:
        train_X, train_y = augment_sequences_gaussian(
            train_X,
            train_y,
            n_aug=aug_n,
            noise_std=aug_noise_std,
        )

    return train_X, train_y, val_X, val_y, test_X, test_y, scaler


# ======================== 5-STEP SEQ2SEQ FORECASTING ========================

def prepare_data_5step_seq2seq(
    dataset: pd.DataFrame,
    pred_column: str,
    lookback: int = 12,
    horizon: int = 5,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.9,
    clip_outliers: bool = False,
    outlier_factor: float = 1.5,
    augment_train: bool = False,
    aug_n: int = 1,
    aug_noise_std: float = 0.01,
    balance_high: bool = False,
    high_quantile: float = 0.8,
    high_repeat: int = 2,
):
    """
    Chuẩn bị dữ liệu cho bài toán (2):
    - Dự báo pred_column cho 5 bước tương lai liên tiếp: t+1..t+5
    - Input: chuỗi dài 'lookback' bước với tất cả features
    - Output: chuỗi dài 'horizon' bước cho pred_column
              (shape: (samples, horizon, 1))

    Thêm:
    - clip_outliers: nếu True, dùng IQR clipping trước khi scale.
    - augment_train: nếu True, thêm nhiễu Gaussian cho train_X.
    - balance_high: nếu True, oversample các mẫu có PM2.5 cao.
    """
    df_proc = dataset.copy()

    # 1) Xử lý outliers (tuỳ chọn)
    if clip_outliers:
        df_proc = clip_outliers_iqr(df_proc, factor=outlier_factor)

    # 2) Scale
    values = df_proc.values.astype("float32")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    pred_idx = df_proc.columns.get_loc(pred_column)

    X_list, y_list = [], []
    n = scaled.shape[0]

    if n <= lookback + horizon:
        raise ValueError("Not enough data for 5-step seq2seq forecasting.")

    # t = chỉ số cuối của input
    # cần t + horizon <= n - 1 => t <= n - horizon - 1
    for t in range(lookback - 1, n - horizon):
        start = t - lookback + 1
        end = t + 1
        X_seq = scaled[start:end, :]  # (lookback, num_features)

        # y: t+1 .. t+horizon
        y_seq = scaled[t + 1: t + 1 + horizon, pred_idx]  # (horizon,)

        X_list.append(X_seq)
        y_list.append(y_seq)

    X = np.array(X_list)                             # (samples, lookback, num_features)
    y = np.array(y_list).reshape(-1, horizon, 1)     # (samples, horizon, 1)

    # 3) Split
    n_samples = X.shape[0]
    n_train = int(n_samples * train_ratio)
    n_valid = int(n_samples * valid_ratio)

    train_X, val_X, test_X = X[:n_train], X[n_train:n_valid], X[n_valid:]
    train_y, val_y, test_y = y[:n_train], y[n_train:n_valid], y[n_valid:]

    # 4) Data augmentation + oversampling cho TRAIN
    if balance_high:
        train_X, train_y = oversample_high_values(
            train_X,
            train_y,
            high_quantile=high_quantile,
            n_repeat=high_repeat,
        )

    if augment_train:
        train_X, train_y = augment_sequences_gaussian(
            train_X,
            train_y,
            n_aug=aug_n,
            noise_std=aug_noise_std,
        )

    return train_X, train_y, val_X, val_y, test_X, test_y, scaler
