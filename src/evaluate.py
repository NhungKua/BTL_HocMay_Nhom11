"""
Mô-đun đánh giá mô hình trên tập Test (Chỉ chạm 1 lần duy nhất).
Thực hiện Thí nghiệm 4: Phân tích các ô nhầm lẫn bằng các điểm láng giềng.
"""
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.neighbors import NearestNeighbors

def main():
    print("\n" + "="*75)
    print("--- ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST (CHẠM 1 LẦN DUY NHẤT) ---")
    print("="*75)

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
    
    print(f"Accuracy trên tập Test:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1 trên tập Test:  {f1:.4f}\n")
    
    print("Báo cáo chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred, digits=4))

    # 5. Tính toán và lưu Ma trận nhầm lẫn vào reports/figures/
    labels = model.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Thật {l}" for l in labels], columns=[f"Đoán {l}" for l in labels])
    print("Ma trận nhầm lẫn (Confusion Matrix):")
    print(cm_df)

    os.makedirs('reports/figures', exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title(f"Ma trận nhầm lẫn trên tập Test (Accuracy = {acc*100:.2f}%)")
    fig_path = 'reports/figures/test_confusion_matrix.png'
    fig.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"\n=> Đã xuất biểu đồ Ma trận nhầm lẫn vào: {fig_path}")
    print("-" * 75)

    # ==========================================
    # THÍ NGHIỆM 4: PHÂN TÍCH LỖI BẰNG LÁNG GIỀNG
    # ==========================================
    print("\n--- THÍ NGHIỆM 4: PHÂN TÍCH CÁC TRƯỜNG HỢP DỰ ĐOÁN SAI ---")
    
    errors_mask = y_test != y_pred
    error_indices = X_test[errors_mask].index
    
    if len(error_indices) == 0:
        print("Mô hình dự đoán đúng 100% trên tập Test. Không có lỗi để phân tích.")
        return

    print(f"Phát hiện {len(error_indices)}/{len(y_test)} trường hợp dự đoán sai trên tập Test.\n")
    
    prep = model.named_steps['preprocessor']
    knn = model.named_steps['classifier']
    
    X_train_transformed = prep.transform(X_train)
    X_test_transformed = prep.transform(X_test)
    
    # Lấy Top 5 láng giềng gần nhất để phân tích toàn diện vùng không gian xung quanh điểm lỗi
    top_k_inspect = max(5, knn.n_neighbors)
    nn = NearestNeighbors(n_neighbors=top_k_inspect, metric=knn.metric)
    nn.fit(X_train_transformed)
    
    num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    
    for idx in error_indices:
        true_label = y_test.iloc[idx]
        pred_label = y_pred[idx]
        print(f"[*] Mẫu Test #{idx}: Nhãn thật = {true_label} | Mô hình đoán = {pred_label} (với K={knn.n_neighbors})")
        print("Số đo hình thái gốc của mẫu bị nhầm lẫn:")
        print(X_test.loc[[idx], num_cols].to_string(index=False))
        
        query = X_test_transformed[idx].reshape(1, -1)
        distances, neighbors_idx = nn.kneighbors(query)
        
        print(f"\nTop {top_k_inspect} láng giềng gần nhất trong tập Train quanh mẫu #{idx}:")
        for i, n_idx in enumerate(neighbors_idx[0]):
            n_label = y_train.iloc[n_idx]
            dist = distances[0][i]
            n_feats = X_train.iloc[n_idx]
            vote_tag = " <-- [Phiếu quyết định K=1]" if (i == 0 and knn.n_neighbors == 1) else ""
            print(
                f"  - Hạng {i+1} (Train #{n_idx:<3}): Loài = {n_label:<9} | "
                f"Khoảng cách ({knn.metric}) = {dist:.4f} | "
                f"Mỏ: {n_feats['bill_length_mm']}x{n_feats['bill_depth_mm']}mm, "
                f"Vây: {n_feats['flipper_length_mm']}mm, Nặng: {n_feats['body_mass_g']}g{vote_tag}"
            )
        print("-" * 75)

if __name__ == "__main__":
    main()