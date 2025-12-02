import os
from pathlib import Path

from src.data_loader import load_dataset
from src.preprocessing import prepare_data_5step_seq2seq
from src.model_builder_seq2seq import build_seq2seq_model
from src.train import create_callbacks, train_model
from src.visualization import (
    visualize_loss,
    visualize_rmse,
    visualize_r2,
    visualize_pm25_comparison,
    visualize_test_metrics,
)
from src.evaluate import evaluate_5step_seq2seq

PRED_COLUMN = "PM2.5"


def main():
    results_dir = Path("results")
    fig_dir = results_dir / "figures_5step"
    ckpt_dir = results_dir / "checkpoints"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # =============== 1+2: LOAD + PREPROCESS ===============
    dataset = load_dataset(data_dir="data")

    (
        train_X,
        train_y,
        val_X,
        val_y,
        test_X,
        test_y,
        scaler,
    ) = prepare_data_5step_seq2seq(
        dataset,
        pred_column=PRED_COLUMN,
        lookback=12,
        horizon=5,
        train_ratio=0.7,
        valid_ratio=0.9,
        # nếu muốn bật xử lý outlier / augmentation / imbalance:
        clip_outliers=True,
        augment_train=True,
        aug_n=2,
        aug_noise_std=0.01,
        balance_high=True,
        high_quantile=0.8,
        high_repeat=2,
    )

    print("Train shape:", train_X.shape, train_y.shape)
    print("Valid shape:", val_X.shape, val_y.shape)
    print("Test shape:", test_X.shape, test_y.shape)

    # =============== 3: BUILD SEQ2SEQ MODEL (LSTM) ===============
    epochs = 50
    batch_size = 64
    learning_rate = 0.001
    horizon = 5

    lookback = train_X.shape[1]
    num_features = train_X.shape[2]

    model = build_seq2seq_model(
        lookback=lookback,
        num_features=num_features,
        horizon=horizon,
        learning_rate=learning_rate,
    )

    # =============== 4: TRAIN ===============
    checkpoint_path = ckpt_dir / "pm25_5step_seq2seq.weights.h5"
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

    # loss / RMSE / R² theo epoch
    visualize_loss(
        history,
        "5-step Seq2Seq Training and Validation Loss",
        save_path=str(fig_dir / "loss_5step_seq2seq.png"),
    )
    visualize_rmse(
        history,
        "5-step Seq2Seq Training and Validation RMSE",
        save_path=str(fig_dir / "rmse_5step_seq2seq.png"),
    )
    visualize_r2(
        history,
        "5-step Seq2Seq Training and Validation R²",
        save_path=str(fig_dir / "r2_5step_seq2seq.png"),
    )

    # =============== 5: EVALUATE ===============
    # evaluate_5step_seq2seq phải được sửa để RETURN:
    #   y_true_5, y_pred_5, (rmse, mae, r2, nse)
    y_true_5, y_pred_5, (rmse, mae, r2, nse) = evaluate_5step_seq2seq(
        model,
        test_X,
        test_y,
        dataset,
        pred_column=PRED_COLUMN,
        results_dir=str(results_dir),
    )

    # ----- 5.1: VẼ BAR 4 METRIC TRÊN TEST -----
    metrics_fig_path = fig_dir / "pm25_5step_seq2seq_test_metrics.png"
    visualize_test_metrics(
        rmse,
        mae,
        r2,
        nse,
        title="5-step Seq2Seq Test Metrics for PM2.5",
        save_path=str(metrics_fig_path),
    )

    # ----- 5.2: VẼ SO SÁNH PM2.5 THỰC VS DỰ ĐOÁN CHO 1 BƯỚC -----
    # y_true_5, y_pred_5 shape: (N, horizon)
    # chọn một bước để vẽ, ví dụ t+1 (index 0) hoặc t+5 (index 4)
    step_to_plot = 4  # 0: t+1, 1: t+2, 2: t+3, 3: t+4, 4: t+5

    compare_fig_path = fig_dir / f"pm25_5step_seq2seq_tplus{step_to_plot+1}_compare.png"
    visualize_pm25_comparison(
        y_true_5[:, step_to_plot],
        y_pred_5[:, step_to_plot],
        n_points=200,
        title=f"PM2.5 (5-step Seq2Seq) - Actual vs Predicted at t+{step_to_plot+1} (first 200 test samples)",
        save_path=str(compare_fig_path),
    )


if __name__ == "__main__":
    main()
