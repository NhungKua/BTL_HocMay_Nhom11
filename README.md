# Project 04: Nhận dạng loài chim cánh cụt bằng K-NN

Dự án kết thúc học phần môn **Học máy cơ bản** - Giảng viên hướng dẫn: **PGS.TS. Nguyễn Văn Hậu**.
Hệ thống ứng dụng thuật toán K-Nearest Neighbors (K-NN) trong Pipeline khép kín (chống rò rỉ dữ liệu) để phân loại 3 loài chim cánh cụt (_Adelie_, _Chinstrap_, _Gentoo_) dựa trên các số đo hình thái học, tích hợp ứng dụng Web FastAPI giải thích kết quả bằng 5 cá thể láng giềng gần nhất và trực quan hóa biên quyết định bằng PCA.

---

## 1. Cấu trúc thư mục dự án

```text
BTL_HocMay_Nhom11/
├── app/
│   ├── main.py                  # Backend FastAPI, REST API (/api/predict) và phục vụ mô hình offline
│   └── templates/               # Giao diện 3 màn hình: intro.html, predict.html, dashboard.html
├── data/
│   ├── penguins.csv             # Dữ liệu gốc tải từ nguồn chính thức
│   └── processed/               # Dữ liệu sau khi chia Stratified Split (70% Train - 15% Val - 15% Test)
├── models/
│   └── best_knn_model.pkl       # Pipeline K-NN hoàn chỉnh đã huấn luyện và đóng băng
├── reports/
│   └── figures/                 # Thư mục lưu biểu đồ EDA và Ma trận nhầm lẫn (Confusion Matrix)
├── src/
│   ├── data.py                  # Tải dữ liệu, làm sạch sơ bộ và chia tập Train/Val/Test
│   ├── features.py              # Định nghĩa Pipeline tiền xử lý (Imputer, StandardScaler, OneHotEncoder)
│   ├── train.py                 # Huấn luyện Baseline, chạy Thí nghiệm 1, Thí nghiệm 2 (GridSearchCV)
│   ├── experiment_3.py          # Chạy Thí nghiệm 3: So sánh bộ đặc trưng số đo vs. thêm biến phân loại
│   └── evaluate.py              # Đánh giá 1 lần trên tập Test & Thí nghiệm 4 (Phân tích lỗi bằng láng giềng)
├── requirements.txt             # Danh sách thư viện phụ thuộc
└── README.md                    # Tài liệu hướng dẫn tái lập và Hồ sơ dữ liệu (Data README)

```

## 2. Hướng dẫn cài đặt và chạy từ môi trường sạch (Reproducibility)

Dự án sử dụng Python 3.9+. Toàn bộ quy trình cố định random_state = 42. Thực hiện tuần tự các lệnh sau trên Terminal:

Bước 1: Tạo và kích hoạt môi trường ảo:
python -m venv venv

# Trên Windows:

venv\Scripts\activate

# Trên macOS/Linux:

source venv/bin/activate

Bước 2: Cài đặt thư viện:

pip install -r requirements.txt

Bước 3: Chạy lại toàn bộ Pipeline Học máy (Tùy chọn kiểm tra tái lập):

python src/data.py # 1. Tải dữ liệu gốc, làm sạch và chia tập 70/15/15
python src/train.py # 2. Chạy Baseline, Thí nghiệm 1, Thí nghiệm 2 và lưu model
python src/experiment_3.py # 3. Chạy Thí nghiệm 3 (So sánh đặc trưng số đo vs. phân loại)
python src/evaluate.py # 4. Đánh giá 1 lần trên tập Test và chạy Thí nghiệm 4 (Phân tích lỗi)

Bước 4: Khởi động Ứng dụng Web:

python app/main.py

## Truy cập ứng dụng trên trình duyệt tại địa chỉ: http://127.0.0.1:8000

Màn hình 1 (Giới thiệu & Phạm vi): http://127.0.0.1:8000/

Màn hình 2 (Dự đoán & 5 Láng giềng): http://127.0.0.1:8000/predict

Màn hình 3 (Dashboard, Model Card & Biên quyết định PCA): http://127.0.0.1:8000/dashboard

## 3. Hồ sơ Dữ liệu (Data README)

3.1. Thông tin chung & Trích dẫn

Tên bộ dữ liệu: Palmer Penguins (penguins.csv)

Nguồn chính thức: https://allisonhorst.github.io/palmerpenguins/

Kho lưu trữ gốc: https://github.com/allisonhorst/palmerpenguins

Ngày tải: 11/10/2026

Phiên bản / Quy mô gốc: 344 dòng (cá thể), 8 cột.

Giấy phép: CC0 (Public Domain) — Ghi công Palmer Station Antarctica LTER và gói palmerpenguins (Horst AM, Hill AP, Gorman KB, 2020).

3.2. Từ điển dữ liệu (Data Dictionary)
| Tên biến | Kiểu dữ liệu | Vai trò | Đơn vị / Ý nghĩa |
| :------------------ | :------------------ | :----------------- | :---------------------------------------- |
| `species` | Phân loại (String) | **Target** (Nhãn) | Tên loài chim (Adelie, Chinstrap, Gentoo) |
| `island` | Phân loại (String) | Feature (Tùy chọn) | Đảo sinh sống (Torgersen, Biscoe, Dream) |
| `bill_length_mm` | Số thực (Float) | Feature (Cốt lõi) | Chiều dài sống mỏ (mm) |
| `bill_depth_mm` | Số thực (Float) | Feature (Cốt lõi) | Độ dày mỏ (mm) |
| `flipper_length_mm` | Số thực (Float) | Feature (Cốt lõi) | Chiều dài vây (mm) |
| `body_mass_g` | Số thực (Float) | Feature (Cốt lõi) | Khối lượng cơ thể (gram) |
| `sex` | Phân loại (String) | Feature (Tùy chọn) | Giới tính (male, female) |
| `year` | Số nguyên (Integer) | Metadata | Năm thu thập dữ liệu (2007, 2008, 2009) |

3.3. Báo cáo chất lượng dữ liệu (Data Quality Report)

- Dữ liệu khuyết (Missing Values):
  Có 02 cá thể (dòng index 3 và 339 trong file gốc) bị khuyết hoàn toàn cả 4 số đo hình thái (bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g) và sex. Quyết định: Loại bỏ 2 dòng này trong src/data.py vì K-NN dựa trên khoảng cách không gian; một điểm mất toàn bộ tọa độ hình thái không có cơ sở thực tế để nội suy. Bộ dữ liệu sạch còn 342 cá thể.

  Có 09 cá thể còn lại chỉ bị khuyết biến sex. Quyết định: Giữ lại toàn bộ 9 dòng này. Khi chạy Thí nghiệm 3, biến sex được điền khuyết bằng SimpleImputer(strategy='most_frequent') đặt bên trong Pipeline (chỉ fit trên Train).

- Dữ liệu trùng lặp & Ngoại lệ (Duplicates & Outliers):

  Kiểm tra trùng lặp: 0 dòng trùng lặp.

  Ngoại lệ: Các số đo nằm trong miền sinh lý tự nhiên của chim cánh cụt trưởng thành (bill_length_mm: 32.1–59.6 mm; bill_depth_mm: 13.1–21.5 mm; flipper_length_mm: 172–231 mm; body_mass_g: 2700–6300 g). Một số cá thể có kích thước lớn/nhỏ hơn mức trung bình của loài là biến thiên sinh học tự nhiên nên được giữ nguyên.

- Phân bố lớp (Class Imbalance):
  Trên 342 cá thể sạch: Adelie chiếm 44.15% (151 cá thể), Gentoo chiếm 35.96% (123 cá thể), Chinstrap chiếm 19.88% (68 cá thể).

  Do phân bố lớp mất cân bằng nhẹ, dự án sử dụng stratify=y khi chia tập (70% Train = 239 mẫu, 15% Val = 51 mẫu, 15% Test = 52 mẫu) và chọn Macro-F1 kết hợp Accuracy làm thước đo chính.

## 4. Tuyên bố sử dụng Công cụ AI & Trách nhiệm Học thuật

Nhóm có sử dụng công cụ AI hỗ trợ gợi ý cấu trúc giao diện HTML/CSS và rà soát cú pháp FastAPI/Scikit-learn. Toàn bộ quy trình chia tập, thiết kế Pipeline chống rò rỉ dữ liệu, chạy thực nghiệm, kiểm chứng số liệu và phân tích lỗi đều do hai thành viên trực tiếp thực thi và kiểm tra chéo.

Ứng dụng Web chỉ phục vụ mục đích học tập và giáo dục sinh thái trong phạm vi quần thể chim cánh cụt tại quần đảo Palmer (Nam Cực), không thay thế giám định sinh học phân tử (DNA).
