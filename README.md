# BTL_HocMay_Nhom11

Penguin species classification using K-NN and machine learning pipelines.

# Project 04: Nhận dạng loài chim cánh cụt bằng K-NN

Dự án cuối kỳ môn Học máy cơ bản. Hệ thống ứng dụng thuật toán K-Nearest Neighbors (K-NN) để phân loại 3 loài chim cánh cụt (Adelie, Gentoo, Chinstrap) dựa trên các số đo hình thái học cơ thể, đi kèm một giao diện Web cho phép dự đoán và giải thích kết quả bằng các láng giềng gần nhất.

## 1. Yêu cầu hệ thống (Prerequisites)

Dự án sử dụng Python 3.9+. Khuyến nghị chạy trong môi trường ảo (Virtual Environment) sạch.
Các thư viện cần thiết được liệt kê trong `requirements.txt`:

- pandas
- scikit-learn
- matplotlib
- seaborn
- fastapi
- uvicorn
- pydantic
- jinja2
- joblib

Cài đặt toàn bộ thư viện bằng lệnh:

```bash
pip install -r requirements.txt
```

## 2. Thông tin chung

- **Nguồn chính thức**: [https://github.com/allisonhorst/palmerpenguins](https://github.com/allisonhorst/palmerpenguins)
- **Ngày tải**: 11/10/2026
- **Tệp sử dụng**: `penguins.csv` (Dữ liệu gốc đã qua làm sạch cơ bản).
- **Giấy phép**: CC0 (Miễn trừ bản quyền) - Ghi công Palmer Station LTER và palmerpenguins.

## 3. Từ điển dữ liệu (Data Dictionary)

Dữ liệu bao gồm 344 cá thể thuộc 3 loài chim cánh cụt (Adelie, Chinstrap, Gentoo) sống trên 3 đảo thuộc quần đảo Palmer.

| Biến (Cột)          | Kiểu dữ liệu       | Đơn vị | Vai trò    | Mô tả                                    |
| :------------------ | :----------------- | :----- | :--------- | :--------------------------------------- |
| `species`           | Phân loại (String) | Không  | **Target** | Loài chim cánh cụt (Đầu ra dự đoán).     |
| `island`            | Phân loại (String) | Không  | Feature    | Đảo nơi cá thể được quan sát (Tùy chọn). |
| `bill_length_mm`    | Số thực (Float)    | mm     | Feature    | Chiều dài mỏ chim.                       |
| `bill_depth_mm`     | Số thực (Float)    | mm     | Feature    | Độ dày mỏ chim.                          |
| `flipper_length_mm` | Số nguyên/Thực     | mm     | Feature    | Chiều dài cánh chèo.                     |
| `body_mass_g`       | Số nguyên/Thực     | gram   | Feature    | Khối lượng cơ thể.                       |
| `sex`               | Phân loại (String) | Không  | Feature    | Giới tính của chim (Tùy chọn).           |
| `year`              | Số nguyên (Int)    | Năm    | ID/Loại    | Năm ghi nhận dữ liệu.                    |
