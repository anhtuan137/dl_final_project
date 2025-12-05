from pathlib import Path
import pandas as pd
from .preprocessing import fillna_back


def load_dataset(data_dir: str = "data") -> pd.DataFrame:
    """
    Đọc tất cả file CSV trong thư mục,
    ghép lại thành 1 DataFrame
    """
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV trong thư mục: {data_dir}")

    dfs = []
    for path in csv_files:
        print("File path:", path)
        dfs.append(pd.read_csv(path, encoding="utf-8"))

    # Ghép tất cả file lại
    df_aqi = pd.concat(dfs, ignore_index=True)

    # Sắp xếp theo thứ tự thời gian từ quá khứ đến hiện tại

    sort_cols = [c for c in ["year", "month", "day", "hour"] if c in df_aqi.columns]
    if sort_cols:
        df_aqi = df_aqi.sort_values(by=sort_cols).reset_index(drop=True)

    # Bỏ các cột không cần thiết (metadata)
    drop_cols = ['No', 'year', 'month', 'day', 'hour', 'wd', 'station']
    drop_cols = [c for c in drop_cols if c in dataset.columns]  # tránh lỗi nếu thiếu cột nào đó
    dataset = dataset.drop(drop_cols, axis=1)

    return dataset
