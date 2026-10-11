"""
Mô-đun định nghĩa Pipeline tiền xử lý dữ liệu.
Đảm bảo mọi bước Imputer, Scaler chỉ fit trên tập Train để chống rò rỉ dữ liệu (Data Leakage).
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(include_categorical=False):
    """
    Xây dựng Pipeline tiền xử lý.
    - include_categorical=False: Chỉ xử lý 4 biến số hình thái (Thí nghiệm 1 & 2)
    - include_categorical=True: Xử lý cả biến số và biến phân loại island, sex (Thí nghiệm 3)
    """
    # 1. Pipeline cho các biến số (Numerical Features)
    num_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    
    # Bước 1.1: Pipeline KHÔNG chuẩn hóa (Dùng cho Thí nghiệm 1 đối chứng)
    num_transformer_unscaled = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')) # Chỉ điền khuyết bằng trung vị, không scale
    ])
    
    # Bước 1.2: Pipeline CÓ chuẩn hóa (Pipeline chính thức)
    num_transformer_scaled = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) # Chuẩn hóa về mean=0, std=1
    ])
    
    # 2. Pipeline cho các biến phân loại (Categorical Features)
    cat_features = ["island", "sex"]
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Điền khuyết giới tính bằng mode
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')) # Mã hóa One-Hot
    ])
    
    # 3. Lắp ráp bằng ColumnTransformer
    if include_categorical:
        # Bộ tiền xử lý cho Thí nghiệm 3 (Full features)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer_scaled, num_features),
                ('cat', cat_transformer, cat_features)
            ])
    else:
        # Bộ tiền xử lý cho Thí nghiệm 1 & 2 (Chỉ dùng biến số)
        preprocessor_scaled = ColumnTransformer(
            transformers=[('num', num_transformer_scaled, num_features)], 
            remainder='drop' # Bỏ qua các cột cat_features
        )
        preprocessor_unscaled = ColumnTransformer(
            transformers=[('num', num_transformer_unscaled, num_features)],
            remainder='drop'
        )
        return preprocessor_scaled, preprocessor_unscaled

    return preprocessor