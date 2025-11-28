import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def nse_score(y_true, y_pred):
    """
    Nash–Sutcliffe Efficiency (NSE)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


def _compute_and_print_metrics(y_true, y_pred, header: str = ""):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nse = nse_score(y_true, y_pred)

    print(header)
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")
    print(f"  R²:   {r2:.3f}")
    print(f"  NSE:  {nse:.3f}")

    return rmse, mae, r2, nse


def evaluate_2step(
    model,
    test_X,
    test_y,
    dataset,
    pred_column: str,
    results_dir: str = "results",
):
    """
    Đánh giá bài toán (1): dự báo t+2
    test_y, model.predict đều ở dạng scaled → unscale bằng min/max của pred_column.
    """
    scaled_pred = model.predict(test_X)        # (N, 1)
    scaled_true = test_y                      # (N, 1)

    max_val = dataset[pred_column].max()
    min_val = dataset[pred_column].min()

    y_true = scaled_true * (max_val - min_val) + min_val
    y_pred = scaled_pred * (max_val - min_val) + min_val

    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)
    metrics_path = os.path.join(results_dir, "logs", "metrics_2step.txt")

    rmse, mae, r2, nse = _compute_and_print_metrics(
        y_true, y_pred, header=f"[2-step forecast for {pred_column}]"
    )

    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(
            f"RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, NSE={nse:.4f}\n"
        )
    return y_true, y_pred, (rmse, mae, r2, nse)

def evaluate_5step_seq2seq(
    model,
    test_X,
    test_y,
    dataset,
    pred_column: str,
    results_dir: str = "results",
):
    scaled_pred = model.predict(test_X)             # (N, H, 1)
    scaled_true = test_y                           # (N, H, 1)

    scaled_pred = np.squeeze(scaled_pred, axis=-1)  # (N, H)
    scaled_true = np.squeeze(scaled_true, axis=-1)  # (N, H)

    max_val = dataset[pred_column].max()
    min_val = dataset[pred_column].min()

    y_true = scaled_true * (max_val - min_val) + min_val
    y_pred = scaled_pred * (max_val - min_val) + min_val

    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)
    metrics_path = os.path.join(results_dir, "logs", "metrics_5step_seq2seq.txt")

    rmse, mae, r2, nse = _compute_and_print_metrics(
        y_true, y_pred, header=f"[5-step seq2seq forecast for {pred_column}]"
    )

    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(
            f"RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, NSE={nse:.4f}\n"
        )
    return y_true, y_pred, (rmse, mae, r2, nse)