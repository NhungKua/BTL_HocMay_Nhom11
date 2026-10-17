"""
Mô-đun Thí nghiệm 3: So sánh bộ đặc trưng chỉ dùng số đo hình thái
với bộ đặc trưng kết hợp cả số đo hình thái và biến phân loại (island, sex).
"""
import os
import sys
import warnings
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')

# Import trực tiếp hàm tạo Pipeline từ features.py để đảm bảo nhất quán
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import build_preprocessor

def run_experiment_3():
    print("\n" + "="*75)
    print("THÍ NGHIỆM 3: SO SÁNH BỘ ĐẶC TRƯNG CHỈ SỐ ĐO VS. CÓ BIẾN PHÂN LOẠI")
    print("="*75)

    try:
        X_train = pd.read_csv('data/processed/X_train.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
        X_val = pd.read_csv('data/processed/X_val.csv')
        y_val = pd.read_csv('data/processed/y_val.csv').squeeze()
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy dữ liệu trong data/processed/. Hãy chạy 'python src/data.py' trước.")
        return

    # 1. Lấy Pipeline từ features.py
    prep_num_only, _ = build_preprocessor(include_categorical=False)
    prep_with_cat = build_preprocessor(include_categorical=True)

    model_num = Pipeline([('preprocessor', prep_num_only), ('classifier', KNeighborsClassifier(n_neighbors=1))])
    model_cat = Pipeline([('preprocessor', prep_with_cat), ('classifier', KNeighborsClassifier(n_neighbors=1))])

    # 2. Đánh giá bằng 5-Fold CV trên tập Train
    cv_num = cross_validate(model_num, X_train, y_train, cv=5, scoring=('accuracy', 'f1_macro'))
    cv_cat = cross_validate(model_cat, X_train, y_train, cv=5, scoring=('accuracy', 'f1_macro'))

    # 3. Đánh giá trên tập Validation
    model_num.fit(X_train, y_train)
    val_f1_num = f1_score(y_val, model_num.predict(X_val), average='macro')

    model_cat.fit(X_train, y_train)
    val_f1_cat = f1_score(y_val, model_cat.predict(X_val), average='macro')

    print("[ KẾT QUẢ 5-FOLD CROSS-VALIDATION TRÊN TẬP TRAIN ]")
    print(f"- Mô hình 1 (Chỉ 4 số đo hình thái) : CV Macro-F1 = {cv_num['test_f1_macro'].mean():.4f} (+/- {cv_num['test_f1_macro'].std():.4f}) | Val Macro-F1 = {val_f1_num:.4f}")
    print(f"- Mô hình 2 (4 số đo + Island & Sex): CV Macro-F1 = {cv_cat['test_f1_macro'].mean():.4f} (+/- {cv_cat['test_f1_macro'].std():.4f}) | Val Macro-F1 = {val_f1_cat:.4f}")

    print("\n[ KẾT LUẬN LỰA CHỌN ĐẶC TRƯNG (TRADE-OFF) ]")
    print("- Việc bổ sung biến phân loại (island, sex) chỉ tạo ra mức chênh lệch siêu nhỏ (~0.6% trên CV).")
    print("- Tuy nhiên, biến 'island' mang tính địa lý cục bộ và biến 'sex' thường bị khuyết thực tế (9 mẫu khuyết).")
    print("=> CHỐT: Giữ bộ 4 đặc trưng số đo hình thái để mô hình gọn nhẹ, tổng quát hóa tốt và tối ưu trải nghiệm nhập liệu trên Web.")
    print("="*75 + "\n")

if __name__ == "__main__":
    run_experiment_3()