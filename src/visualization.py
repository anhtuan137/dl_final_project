import os
import math
import matplotlib.pyplot as plt
from pylab import mpl
import seaborn as sns
from pandas import DataFrame
from typing import Sequence, Optional
import numpy as np

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 7)
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["axes.unicode_minus"] = False

titles = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEM", "PRES", "DEMP", "RAIN", "WSPM",
]

feature_keys = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEM", "PRES", "DEMP", "RAIN", "WSPM",
]


def visualize_loss(history, title: str = "Training and Validation Loss", save_path: Optional[str] = None):
    """Vẽ loss và val_loss theo epoch."""
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))

    plt.figure(dpi=300)
    plt.plot(epochs, loss, "green", label="Training Loss")
    plt.plot(epochs, val_loss, color="purple", label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def visualize_rmse(history, title: str = "Training and Validation RMSE", save_path: Optional[str] = None):
    """
    Vẽ RMSE và val_RMSE theo epoch.

    Cần: model.compile(..., metrics=[rmse, ...]) từ src.metrics
    => history.history có 'rmse' và 'val_rmse'.
    """
    rmse = history.history["rmse"]
    val_rmse = history.history["val_rmse"]
    epochs = range(len(rmse))

    plt.figure(dpi=300)
    plt.plot(epochs, rmse, "b", label="Training RMSE")
    plt.plot(epochs, val_rmse, "r", label="Validation RMSE")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def visualize_r2(history, title: str = "Training and Validation R²", save_path: Optional[str] = None):
    """
    Vẽ R² và val_R² theo epoch.

    Cần: model.compile(..., metrics=[..., r2]) từ src.metrics
    => history.history có 'r2' và 'val_r2'.
    """
    r2 = history.history["r2"]
    val_r2 = history.history["val_r2"]
    epochs = range(len(r2))

    plt.figure(dpi=300)
    plt.plot(epochs, r2, "b", label="Training R²")
    plt.plot(epochs, val_r2, "r", label="Validation R²")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("R²")
    plt.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def compare_visual(
    data1: DataFrame,
    data2: DataFrame,
    save_path: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
    ncols: int = 3,
):
    """
    So sánh actual vs prediction cho nhiều biến (nếu bạn còn dùng).
    """
    if columns is None:
        common_cols = [c for c in feature_keys if c in data1.columns and c in data2.columns]
        if not common_cols:
            common_cols = list(data1.columns.intersection(data2.columns))
        columns = common_cols

    if len(columns) == 0:
        raise ValueError("Không tìm thấy cột chung nào giữa data1 và data2 để vẽ.")

    n_plots = len(columns)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4 * nrows),
        dpi=300,
        facecolor="w",
        edgecolor="k",
    )

    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.ravel()

    for i, key in enumerate(columns):
        ax = axes[i]
        data1[key].plot(
            ax=ax,
            rot=25,
            label="actual",
        )
        data2[key].plot(
            ax=ax,
            color="red",
            rot=25,
            label="prediction",
        )

        if key in titles:
            ax.set_title(key)
        else:
            ax.set_title(str(key))

        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def visualize_pm25_comparison(
    y_true,
    y_pred,
    n_points: int = 200,
    title: str = "PM2.5: Actual vs Predicted (first N test samples)",
    save_path: Optional[str] = None,
):
    """
    Vẽ đường so sánh PM2.5 thực tế vs dự đoán trên test.
    y_true, y_pred: array-like (N,) hoặc (N,1)
    n_points: số điểm đầu tiên trên test để vẽ (để hình không quá dài).
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    n_points = min(n_points, len(y_true))

    plt.figure(dpi=300, figsize=(10, 4))
    plt.plot(range(n_points), y_true[:n_points], label="Actual PM2.5")
    plt.plot(range(n_points), y_pred[:n_points], label="Predicted PM2.5", alpha=0.8)
    plt.xlabel("Sample index (trên tập test)")
    plt.ylabel("PM2.5")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def visualize_test_metrics(
    rmse: float,
    mae: float,
    r2: float,
    nse: float,
    title: str = "Test Metrics",
    save_path: Optional[str] = None,
):
    """
    Vẽ biểu đồ bar 4 metric: RMSE, MAE, R², NSE trên tập test.
    """
    metrics = ["RMSE", "MAE", "R²", "NSE"]
    values = [rmse, mae, r2, nse]

    plt.figure(dpi=300, figsize=(6, 4))
    bars = plt.bar(metrics, values)
    plt.title(title)

    # Ghi số lên trên cột
    for bar, v in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{v:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
