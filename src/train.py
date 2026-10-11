"""
Mô-đun huấn luyện mô hình, bao gồm Baseline và các Thí nghiệm 1 & 2.
"""
import os
import sys
import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

# Add current dir to path to import features.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import build_preprocessor

def evaluate_and_print(name, model, X_train, y_train, X_val, y_val):
    """Hàm phụ trợ để huấn luyện, dự đoán và in kết quả"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f"{name:<45} | Acc: {acc:.4f} | F1 Macro: {f1:.4f}")
    return model, acc, f1

def main():
    # 1. Đọc dữ liệu
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_val = pd.read_csv('data/processed/y_val.csv').squeeze()

    print("\n" + "="*70)
    print("--- ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP VALIDATION ---")
    print("="*70)

    # ==========================================
    # BASELINE MODEL
    # ==========================================
    dummy = DummyClassifier(strategy='prior', random_state=42)
    evaluate_and_print("[Baseline] Dummy Classifier", dummy, X_train, y_train, X_val, y_val)
    print("-" * 70)

    # Lấy các Pipeline tiền xử lý từ features.py
    prep_scaled, prep_unscaled = build_preprocessor(include_categorical=False)

    # ==========================================
    # THÍ NGHIỆM 1: KHÔNG CHUẨN HÓA VS. CÓ CHUẨN HÓA
    # ==========================================
    # Cấu hình K-NN cơ bản với K=5
    knn_base = KNeighborsClassifier(n_neighbors=5)

    # 1.1 K-NN Không chuẩn hóa
    pipe_unscaled = Pipeline(steps=[('preprocessor', prep_unscaled), ('classifier', knn_base)])
    evaluate_and_print("[Exp 1] K-NN (K=5) - KHÔNG chuẩn hóa", pipe_unscaled, X_train, y_train, X_val, y_val)

    # 1.2 K-NN Có chuẩn hóa
    pipe_scaled = Pipeline(steps=[('preprocessor', prep_scaled), ('classifier', knn_base)])
    evaluate_and_print("[Exp 1] K-NN (K=5) - CÓ chuẩn hóa", pipe_scaled, X_train, y_train, X_val, y_val)
    print("-" * 70)

    # ==========================================
    # THÍ NGHIỆM 2: TÌM K TỐT NHẤT BẰNG CROSS-VALIDATION
    # ==========================================
    print("Đang chạy Grid Search CV trên tập Train để tìm K tối ưu...")
    # Khai báo không gian siêu tham số cần dò tìm
    param_grid = {
        'classifier__n_neighbors': [1, 3, 5, 7, 9, 11, 15],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan']
    }

    # Đưa pipeline ĐÃ CHUẨN HÓA vào GridSearchCV
    grid_search = GridSearchCV(
        pipe_scaled, 
        param_grid, 
        cv=5, # Cross-validation 5 fold
        scoring='f1_macro', 
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"\n=> Tham số K-NN tốt nhất tìm được: {best_params}")
    evaluate_and_print(f"[Exp 2] K-NN Tối ưu", best_model, X_train, y_train, X_val, y_val)
    print("="*70 + "\n")

    # Lưu mô hình tốt nhất lại để dùng cho Web API sau này
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_knn_model.pkl')
    print("Đã lưu mô hình tốt nhất vào thư mục models/best_knn_model.pkl")

if __name__ == "__main__":
    main()