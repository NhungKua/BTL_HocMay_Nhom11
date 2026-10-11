from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ConfigDict
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg') # Cần thiết để vẽ biểu đồ không cần GUI
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io
import base64
import uvicorn
import os

app = FastAPI(
    title="Palmer Penguins API",
    description="API phân loại chim cánh cụt bằng K-NN. Cung cấp Dashboard và Model Card.",
    version="1.0.0"
)

# Thiết lập template engine
os.makedirs('app/templates', exist_ok=True)
templates = Jinja2Templates(directory="app/templates")

# Tải mô hình và dữ liệu Train/Val (offline serving)
try:
    model = joblib.load('models/best_knn_model.pkl')
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_val = pd.read_csv('data/processed/y_val.csv').squeeze()
    
    # Trích xuất các thành phần từ pipeline đã huấn luyện
    prep = model.named_steps['preprocessor']
    knn_classifier = model.named_steps['classifier']
    
    # Chuẩn bị NearestNeighbors để giải thích top 5
    X_train_transformed = prep.transform(X_train)
    nn_explainer = NearestNeighbors(n_neighbors=5, metric=knn_classifier.metric)
    nn_explainer.fit(X_train_transformed)
    
except FileNotFoundError:
    print("WARNING: Chưa tìm thấy mô hình hoặc dữ liệu Train/Val. Hãy chạy src/train.py trước.")
    model = None

# Schema kiểm tra miền giá trị đầu vào (Validation)
class PenguinData(BaseModel):
    bill_length_mm: float = Field(..., ge=20.0, le=70.0, description="Độ dài mỏ (20-70 mm)")
    bill_depth_mm: float = Field(..., ge=10.0, le=25.0, description="Độ dày mỏ (10-25 mm)")
    flipper_length_mm: float = Field(..., ge=150.0, le=250.0, description="Độ dài vây (150-250 mm)")
    body_mass_g: float = Field(..., ge=2000.0, le=7000.0, description="Khối lượng (2000-7000 g)")

    # Khai báo model_config theo chuẩn Pydantic V2
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bill_length_mm": 45.0,
                "bill_depth_mm": 15.0,
                "flipper_length_mm": 210.0,
                "body_mass_g": 4500.0
            }
        }
    )

# --- API ENDPOINTS ---

@app.post("/api/predict")
def predict_penguin(data: PenguinData):
    """API Endpoint nhận số đo và trả về dự đoán cùng 5 láng giềng."""
    if model is None:
        raise HTTPException(status_code=500, detail="Hệ thống chưa tải được mô hình.")
    
    # Tạo DataFrame từ input, bù cột giả cho island và sex (vì preprocessor sẽ drop chúng)
    input_df = pd.DataFrame([{
        "bill_length_mm": data.bill_length_mm,
        "bill_depth_mm": data.bill_depth_mm,
        "flipper_length_mm": data.flipper_length_mm,
        "body_mass_g": data.body_mass_g,
        "island": "Biscoe", 
        "sex": "male"       
    }])

    # 1. Dự đoán
    prediction = model.predict(input_df)[0]
    actual_k = knn_classifier.n_neighbors
    
    # 2. Tìm 5 láng giềng gần nhất
    query_transformed = prep.transform(input_df)
    distances, indices = nn_explainer.kneighbors(query_transformed)
    
    neighbors = []
    votes_in_top_5 = 0
    
    for i, idx in enumerate(indices[0]):
        sp = str(y_train.iloc[idx])
        if sp == prediction:
            votes_in_top_5 += 1
            
        neighbors.append({
            "rank": i + 1,
            "species": sp,
            "distance": round(float(distances[0][i]), 4),
            "bill_length_mm": float(X_train.iloc[idx]['bill_length_mm']),
            "bill_depth_mm": float(X_train.iloc[idx]['bill_depth_mm']),
            "flipper_length_mm": float(X_train.iloc[idx]['flipper_length_mm']),
            "body_mass_g": float(X_train.iloc[idx]['body_mass_g'])
        })
        
    # 3. Tính độ đồng thuận
    consensus_confidence = round((votes_in_top_5 / 5) * 100, 2)
    
    # 4. Sinh lý do
    if actual_k == 1:
        reason = f"Thuật toán tối ưu đang dùng K=1. Vì láng giềng gần nhất (khoảng cách {neighbors[0]['distance']:.4f}) là loài {prediction}, mô hình chốt dự đoán với độ tin cậy gốc 100%. Khi mở rộng quan sát Top 5 láng giềng, loài {prediction} chiếm {votes_in_top_5}/5 cá thể."
    else:
        reason = f"Mô hình sử dụng K={actual_k}. Trong top 5 láng giềng gần nhất, loài {prediction} chiếm {votes_in_top_5}/5 phiếu."

    return {
        "status": "success",
        "predicted_species": prediction,
        "confidence_percent": consensus_confidence,
        "reasoning": reason,
        "neighbors": neighbors
    }

@app.get("/api/boundary")
def get_decision_boundary(k: int = 1):
    """API sinh hình ảnh biên quyết định K-NN bằng PCA (Giảm chiều 2D)"""
    try:
        X = prep.transform(X_train)
        y = y_train.map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}).values
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        knn_2d = KNeighborsClassifier(n_neighbors=k, metric=knn_classifier.metric, weights=knn_classifier.weights)
        knn_2d.fit(X_pca, y)
        
        h = .02  
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Dùng fig, ax để chống lỗi ghi đè đồ họa đa luồng
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ['#FF0000', '#00FF00', '#0000FF']
        
        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
        
        for i, color in zip([0, 1, 2], cmap_bold):
            ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], c=color, label=['Adelie', 'Chinstrap', 'Gentoo'][i], edgecolor='k', s=30)
            
        ax.set_title(f"Biên quyết định K-NN (K={k}) trên 2 thành phần PCA chính")
        ax.set_xlabel("PCA Thành phần 1")
        ax.set_ylabel("PCA Thành phần 2")
        ax.legend()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return {"status": "success", "image_base64": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/species_distribution")
def get_species_distribution():
    """API sinh hình ảnh biểu đồ tròn phân bố loài (dữ liệu thô)"""
    try:
        df = pd.read_csv('data/penguins.csv').dropna(subset=["bill_length_mm"])
        counts = df['species'].value_counts()
        
        # Dùng fig, ax
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, 
                colors=['#ff9999','#66b3ff','#99ff99'], wedgeprops={'edgecolor': 'black'})
        ax.set_title("Tỷ lệ phân bố các loài trong bộ dữ liệu", fontweight='bold')
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return {"status": "success", "image_base64": base64.b64encode(buf.read()).decode('utf-8')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/k_metrics")
def get_k_metrics():
    """API tính toán 4 chỉ số bằng Kiểm định chéo (5-Fold CV) trên tập Train"""
    try:
        X_train_tf = prep.transform(X_train)
        
        results = []
        for k in range(1, 16, 2): 
            knn = KNeighborsClassifier(n_neighbors=k, metric=knn_classifier.metric, weights=knn_classifier.weights)
            
            # Chạy kiểm định chéo 5-fold mô phỏng GridSearchCV
            scores = cross_validate(
                knn, X_train_tf, y_train, cv=5, 
                scoring=('accuracy', 'precision_macro', 'recall_macro', 'f1_macro')
            )
            
            # Lấy trung bình 5 lần chạy
            acc = scores['test_accuracy'].mean()
            prec = scores['test_precision_macro'].mean()
            rec = scores['test_recall_macro'].mean()
            f1 = scores['test_f1_macro'].mean()
            
            results.append({
                "k": k, 
                "accuracy": round(acc, 4), 
                "precision": round(prec, 4),
                "recall": round(rec, 4), 
                "f1": round(f1, 4)
            })
        return {"status": "success", "metrics": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- WEB ROUTES (3 Màn hình) ---

@app.get("/", response_class=HTMLResponse)
def intro_page(request: Request):
    """Màn hình 1: Giới thiệu và phạm vi"""
    return templates.TemplateResponse(request=request, name="intro.html")

@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    """Màn hình 2: Thao tác chính (Form nhập liệu)"""
    return templates.TemplateResponse(request=request, name="predict.html")

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    """Màn hình 3: Dashboard đánh giá & Model Card"""
    return templates.TemplateResponse(request=request, name="dashboard.html")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)