"""
Mô-đun huấn luyện mô hình và thiết lập Baseline.
"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

def run_baseline():
    """
    Chạy mô hình DummyClassifier (luôn đoán lớp phổ biến nhất) để làm Baseline.
    """
    print("\n" + "="*50)
    print("--- CHẠY MÔ HÌNH BASELINE (DUMMY CLASSIFIER) ---")
    print("="*50)
    
    # 1. Đọc lại tập Train và Val
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_val = pd.read_csv('data/processed/y_val.csv').squeeze()
    
    # 2. Khởi tạo và huấn luyện Dummy Classifier
    # 'prior' nghĩa là luôn dự đoán lớp xuất hiện nhiều nhất trong tập Train
    dummy_clf = DummyClassifier(strategy='prior', random_state=42)
    dummy_clf.fit(X_train, y_train)
    
    # 3. Dự đoán trên tập Validation (ĐẢM BẢO KHÔNG DÙNG TẬP TEST Ở ĐÂY)
    y_val_pred = dummy_clf.predict(X_val)
    
    # 4. Tính toán độ đo
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, average='macro') 
    
    print(f"Độ chính xác (Accuracy) trên Validation: {acc:.4f}")
    print(f"Macro F1-Score trên Validation:          {f1:.4f}")
    print("\n=> LƯU Ý: Bất kỳ mô hình K-NN nào cũng phải vượt qua cột mốc này.")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_baseline()