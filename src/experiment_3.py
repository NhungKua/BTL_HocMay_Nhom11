"""
Mô-đun Thí nghiệm 3: So sánh bộ đặc trưng chỉ dùng số đo hình thái
với bộ đặc trưng kết hợp cả số đo hình thái và biến phân loại (island, sex).
"""
import pandas as pd
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

warnings.filterwarnings('ignore')

def run_experiment_3():
    print("\n" + "="*70)
    print("THÍ NGHIỆM 3: SO SÁNH HIỆU NĂNG THEO BỘ ĐẶC TRƯNG (FEATURE SELECTION)")
    print("="*70)

    # 1. Tải và chuẩn bị lại dữ liệu thô (tương tự như src/data.py)
    print("Đang tải dữ liệu gốc từ thư viện...")
    df = sns.load_dataset("penguins")
    
    # Làm sạch dòng khuyết 4 số đo
    df_clean = df.dropna(subset=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']).copy()
    
    # Tách X, y
    X = df_clean.drop(columns=['species'])
    y = df_clean['species']
    
    # Chia tập Train y như cũ để đảm bảo công bằng (Chỉ lấy tập Train để chạy CV)
    X_train_all, _, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    # Trích xuất riêng DataFrame chỉ chứa 4 số đo để chạy Mô hình 1
    num_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X_train_num = X_train_all[num_features]

    # --- KHỞI TẠO MÔ HÌNH 1: CHỈ DÙNG 4 SỐ ĐO HÌNH THÁI ---
    prep_num = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    model_1 = Pipeline([('prep', prep_num), ('knn', KNeighborsClassifier(n_neighbors=1))])

    # --- KHỞI TẠO MÔ HÌNH 2: 4 SỐ ĐO + 2 BIẾN PHÂN LOẠI ---
    cat_features = ['island', 'sex']
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore'))    
    ])
    
    # Kết hợp 2 bộ tiền xử lý
    prep_all = ColumnTransformer(transformers=[
        ('num', prep_num, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    model_2 = Pipeline([('prep', prep_all), ('knn', KNeighborsClassifier(n_neighbors=1))])

    # --- CHẠY KIỂM ĐỊNH CHÉO (5-FOLD CV) TRÊN TẬP TRAIN ---
    print("Đang chạy 5-Fold Cross Validation trên tập Train...\n")
    cv_1 = cross_validate(model_1, X_train_num, y_train, cv=5, scoring='f1_macro')
    cv_2 = cross_validate(model_2, X_train_all, y_train, cv=5, scoring='f1_macro')

    f1_model_1 = cv_1['test_score'].mean()
    f1_model_2 = cv_2['test_score'].mean()

    # --- IN KẾT QUẢ VÀ KẾT LUẬN ---
    print("[ KẾT QUẢ SO SÁNH ]")
    print(f"- Mô hình 1 (Chỉ 4 Số đo hình thái) : F1-Macro = {f1_model_1:.4f}")
    print(f"- Mô hình 2 (Số đo + Island & Sex): F1-Macro = {f1_model_2:.4f}")
    
    print("\n[ ĐÁNH GIÁ TRADE-OFF ]")
    if f1_model_2 > f1_model_1:
        diff = (f1_model_2 - f1_model_1) * 100
        print(f"Mô hình 2 có điểm cao hơn Mô hình 1 khoảng {diff:.2f}%.")
    else:
        print("Mô hình 1 có điểm số tốt hơn hoặc bằng Mô hình 2.")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    run_experiment_3()