"""
Mô-đun đánh giá mô hình trên tập Test (Chỉ chạm 1 lần duy nhất).
Thực hiện Thí nghiệm 4: Phân tích các ô nhầm lẫn bằng các điểm láng giềng.
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import NearestNeighbors

def main():
    print("\n" + "="*70)
    print("--- ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST (CHẠM 1 LẦN DUY NHẤT) ---")
    print("="*70)

    # 1. Đọc tập Test (để đánh giá) và Train (để truy vết láng giềng)
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()

    # 2. Tải mô hình tốt nhất từ Thí nghiệm 2
    try:
        model = joblib.load('models/best_knn_model.pkl')
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy mô hình. Hãy chạy src/train.py trước.")
        return

    # 3. Dự đoán trên Test
    y_pred = model.predict(X_test)

    # 4. Báo cáo chỉ số cốt lõi
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy trên tập Test:  {acc:.4f}")
    print(f"Macro F1 trên tập Test:  {f1:.4f}\n")
    
    print("Báo cáo chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred))

    # 5. Tính toán Ma trận nhầm lẫn
    labels = model.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Thật {l}" for l in labels], columns=[f"Đoán {l}" for l in labels])
    print("Ma trận nhầm lẫn (Confusion Matrix):")
    print(cm_df)
    print("-" * 70)

    # ==========================================
    # THÍ NGHIỆM 4: PHÂN TÍCH LỖI BẰNG LÁNG GIỀNG
    # ==========================================
    print("\n--- THÍ NGHIỆM 4: PHÂN TÍCH CÁC TRƯỜNG HỢP DỰ ĐOÁN SAI ---")
    
    # Tìm các vị trí dự đoán sai
    errors_mask = y_test != y_pred
    error_indices = X_test[errors_mask].index
    
    if len(error_indices) == 0:
        print("Tuyệt vời! Mô hình dự đoán đúng 100% trên tập Test. Không có lỗi để phân tích.")
        return

    print(f"Phát hiện {len(error_indices)} trường hợp dự đoán sai.\n")
    
    # Trích xuất Pipeline tiền xử lý và Mô hình K-NN từ best_model
    prep = model.named_steps['preprocessor']
    knn = model.named_steps['classifier']
    
    # Transform dữ liệu để tính toán khoảng cách đúng như thuật toán đã học
    X_train_transformed = prep.transform(X_train)
    X_test_transformed = prep.transform(X_test)
    
    # Khởi tạo thuật toán tìm láng giềng độc lập để xuất khoảng cách
    nn = NearestNeighbors(n_neighbors=knn.n_neighbors, metric=knn.metric)
    nn.fit(X_train_transformed)
    
    for idx in error_indices:
        true_label = y_test.iloc[idx]
        pred_label = y_pred[idx]
        print(f"[*] Mẫu Test #{idx}: Thật = {true_label} | Đoán = {pred_label}")
        print("Đặc trưng gốc của mẫu này:")
        print(X_test.iloc[idx].to_frame().T.to_string(index=False))
        
        # Truy vết các láng giềng đã tham gia bỏ phiếu
        query = X_test_transformed[idx].reshape(1, -1)
        distances, neighbors_idx = nn.kneighbors(query)
        
        print(f"\n{knn.n_neighbors} láng giềng gần nhất trong tập Train đã bỏ phiếu:")
        for i, n_idx in enumerate(neighbors_idx[0]):
            n_label = y_train.iloc[n_idx]
            dist = distances[0][i]
            print(f"  - Láng giềng {i+1} (ID Train: {n_idx}): Nhãn = {n_label:<10} | Khoảng cách {knn.metric} = {dist:.4f}")
        print("-" * 70)

if __name__ == "__main__":
    main()