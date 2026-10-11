"""
Mô-đun chuẩn bị dữ liệu Palmer Penguins:
1. Tải dữ liệu từ GitHub.
2. Lưu dữ liệu gốc.
3. Làm sạch sơ bộ (loại các dòng khuyết số đo hình thái).
4. Chia Train / Validation / Test theo phân tầng.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Đường dẫn và nguồn dữ liệu
DATA_URL = (
    "https://raw.githubusercontent.com/allisonhorst/"
    "palmerpenguins/main/inst/extdata/penguins.csv"
)

RAW_DATA_PATH = "data/penguins.csv"
PROCESSED_DIR = "data/processed"

# Lấy thêm island và sex để phục vụ Thí nghiệm 3 sau này. 
# Cột 'year' bị loại vì không cần thiết cho dự đoán sinh học.
FEATURES_COLS = [
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex"
]

def download_data(url=DATA_URL, output_path=RAW_DATA_PATH):
    """Tải dữ liệu CSV từ GitHub và lưu vào dự án."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Đang tải dữ liệu Palmer Penguins...")
    df = pd.read_csv(url)
    df.to_csv(output_path, index=False)
    
    print(f"Tải thành công: {output_path}")
    print(f"Số dòng dữ liệu gốc: {len(df)}")
    return df

def load_and_split_data(data_path=RAW_DATA_PATH, random_state=42):
    """Làm sạch sơ bộ và chia dữ liệu theo tỷ lệ 70/15/15."""
    # 1. Đọc dữ liệu gốc
    df = pd.read_csv(data_path)

    # 2. Loại các dòng thiếu ít nhất một trong bốn số đo hình thái
    # Lưu ý: Vẫn giữ lại dòng nếu chỉ thiếu 'sex'.
    num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    df_cleaned = df.dropna(subset=num_cols).copy()

    print(f"Số dòng ban đầu: {len(df)}")
    print(f"Số dòng sau làm sạch: {len(df_cleaned)}")
    print(f"Số dòng bị loại: {len(df) - len(df_cleaned)}")

    # 3. Tách Features (X) và Target (y)
    X = df_cleaned[FEATURES_COLS]
    y = df_cleaned["species"]

    # 4. Chia 70% Train và 30% còn lại (áp dụng phân tầng để cân bằng lớp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    # 5. Chia phần còn lại thành 15% Validation và 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    # 6. Tạo thư mục và lưu các tập dữ liệu
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    datasets = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    for name, data in datasets.items():
        data.to_csv(
            os.path.join(PROCESSED_DIR, f"{name}.csv"),
            index=False,
        )

    print("\nĐã chia và lưu dữ liệu thành công.")
    print(f"Train: {len(X_train)} dòng")
    print(f"Validation: {len(X_val)} dòng")
    print(f"Test: {len(X_test)} dòng")

    # Báo cáo tỷ lệ loài trong từng tập để chứng minh Stratify hoạt động đúng
    for name, target in [
        ("Train", y_train),
        ("Validation", y_val),
        ("Test", y_test),
    ]:
        print(f"\nTỷ lệ loài trong tập {name}:")
        print((target.value_counts(normalize=True) * 100).round(2))

    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    """Chạy toàn bộ quy trình chuẩn bị dữ liệu."""
    # Tải dữ liệu trước
    download_data()
    
    # Sau đó làm sạch và chia tập
    print("\n--- TIẾN HÀNH CHIA TẬP DỮ LIỆU ---")
    load_and_split_data()

if __name__ == "__main__":
    main()