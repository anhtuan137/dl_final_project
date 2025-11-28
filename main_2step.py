import os
from pathlib import Path

from src.data_loader import load_dataset
from src.preprocessing import prepare_data_2step
from src.model_builder_2step import build_2step_model
from src.train import create_callbacks, train_model
from src.visualization import (
    visualize_loss,
    visualize_rmse,
    visualize_r2,
    visualize_pm25_comparison,
    visualize_test_metrics,
)
from src.evaluate import evaluate_2step

PRED_COLUMN = "PM2.5"


def main():
    # ================== 0. PATHS ==================
    results_dir = Path("results")
    fig_dir = results_dir / "figures_2step"
    ckpt_dir = results_dir / "checkpoints"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ================== 1. LOAD DATA ==================
    dataset = load_dataset(data_dir="data")

    # ================== 2. PREPROCESS (2-STEP + XỬ LÝ DỮ LIỆU) ==================
    # Bật các option:
    #   - clip_outliers=True: cắt ngoại lai theo IQR
    #   - augment_train=True: thêm noise Gaussian vào chuỗi train (data augmentation)
    #   - balance_high=True: oversample các mẫu có PM2.5 cao
    train_X, train_y, val_X, val_y, test_X, test_y, scaler = prepare_data_2step(
        dataset,
        pred_column=PRED_COLUMN,
        lookback=12,
        train_ratio=0.7,
        valid_ratio=0.9,
        clip_outliers=True,      # xử lý ngoại lai
        outlier_factor=1.5,
        #augment_train=True,      # làm giàu dữ liệu train bằng nhiễu nhỏ
        #aug_n=2,                 # mỗi mẫu train nhân thêm 2 bản noisy
        aug_noise_std=0.01,
        #balance_high=True,       # oversample mẫu PM2.5 cao
        high_quantile=0.8,
        #high_repeat=2,
    )

    print("Train shape:", train_X.shape, train_y.shape)
    print("Valid shape:", val_X.shape, val_y.shape)
    print("Test shape:", test_X.shape, test_y.shape)

    # ================== 3. BUILD MODEL (SimpleRNN 2-STEP) ==================
    epochs = 2
    batch_size = 128
    learning_rate = 0.001

    lookback = train_X.shape[1]
    num_features = train_X.shape[2]

    model = build_2step_model(
        lookback=lookback,
        num_features=num_features,
        learning_rate=learning_rate,
    )

    # ================== 4. TRAIN ==================
    checkpoint_path = ckpt_dir / "pm25_2step.weights.h5"
    callbacks = create_callbacks(str(checkpoint_path), patience=10)

    history = train_model(
        model,
        train_X,
        train_y,
        val_X,
        val_y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    # ====== VẼ LOSS / RMSE / R² THEO EPOCH ======
    visualize_loss(
        history,
        "2-step Training and Validation Loss",
        save_path=str(fig_dir / "loss_2step.png"),
    )

    visualize_rmse(
        history,
        "2-step Training and Validation RMSE",
        save_path=str(fig_dir / "rmse_2step.png"),
    )

    visualize_r2(
        history,
        "2-step Training and Validation R²",
        save_path=str(fig_dir / "r2_2step.png"),
    )

    # ================== 5. EVALUATE ==================
    # 5: evaluate
    y_true, y_pred, (rmse, mae, r2, nse) = evaluate_2step(
        model,
        test_X,
        test_y,
        dataset,
        pred_column=PRED_COLUMN,
        results_dir=str(results_dir),
    )

    # 5.1: Vẽ so sánh PM2.5 thực vs dự đoán trên test
    compare_fig_path = fig_dir / "pm25_2step_compare_first200.png"
    visualize_pm25_comparison(
        y_true,
        y_pred,
        n_points=200,
        title="PM2.5 (2-step) - Actual vs Predicted (first 200 test samples)",
        save_path=str(compare_fig_path),
    )

    # 5.2: Vẽ bar chart 4 metrics trên test
    metrics_fig_path = fig_dir / "pm25_2step_test_metrics.png"
    visualize_test_metrics(
        rmse,
        mae,
        r2,
        nse,
        title="2-step Test Metrics for PM2.5",
        save_path=str(metrics_fig_path),
    )
   

if __name__ == "__main__":
    main()
