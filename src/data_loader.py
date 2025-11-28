from pathlib import Path
import pandas as pd
from .preprocessing import fillna_back


def load_dataset(data_dir: str = "data") -> pd.DataFrame:
    """
    Đọc tất cả file CSV trong thư mục data_dir, ghép lại thành 1 DataFrame
    và xử lý missing values + drop các cột không dùng.
    """
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV trong thư mục: {data_dir}")

    dfs = []
    for path in csv_files:
        print("File path:", path)
        dfs.append(pd.read_csv(path, encoding="utf-8"))

    df_aqi = pd.concat(dfs, ignore_index=True)

    # Điền giá trị thiếu
    dataset = fillna_back(df_aqi)

    # Bỏ các cột không cần
    dataset = dataset.drop(
        ['No', 'year', 'month', 'day', 'hour', 'wd', 'station'],
        axis=1
    )

    return dataset
