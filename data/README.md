# Hồ Sơ Dữ Liệu: Palmer Penguins

Tài liệu này cung cấp thông tin chi tiết về bộ dữ liệu được sử dụng trong **Project 04: Nhận dạng loài chim cánh cụt bằng K-NN**, bao gồm nguồn gốc, từ điển dữ liệu và quy trình làm sạch.

## 1. Nguồn dữ liệu và Quyền sử dụng

- **Tên bộ dữ liệu:** Palmer Penguins
- **Nguồn cung cấp:** Thu thập tại Trạm Palmer LTER, Nam Cực (2007-2009). Được phân phối công khai thông qua kho lưu trữ GitHub của Allison Horst.
- **Quyền sử dụng:** Phạm vi công cộng (CC0 - Creative Commons Zero).
- **Trích dẫn (Citation):** Gorman KB, Williams TD, Fraser WR (2014) Ecological Sexual Dimorphism and Environmental Variability within a Community of Antarctic Penguins (Genus _Pygoscelis_). PLoS ONE 9(3): e90081. doi:10.1371/journal.pone.0090081.

## 2. Từ điển dữ liệu (Data Dictionary)

Bộ dữ liệu gốc gồm 344 quan sát với 8 biến (cột).

| Tên biến            | Kiểu dữ liệu        | Vai trò            | Đơn vị / Ý nghĩa                          |
| :------------------ | :------------------ | :----------------- | :---------------------------------------- |
| `species`           | Phân loại (String)  | **Target** (Nhãn)  | Tên loài chim (Adelie, Chinstrap, Gentoo) |
| `island`            | Phân loại (String)  | Feature (Tùy chọn) | Đảo sinh sống (Torgersen, Biscoe, Dream)  |
| `bill_length_mm`    | Số thực (Float)     | Feature (Cốt lõi)  | Chiều dài sống mỏ (mm)                    |
| `bill_depth_mm`     | Số thực (Float)     | Feature (Cốt lõi)  | Độ dày mỏ (mm)                            |
| `flipper_length_mm` | Số thực (Float)     | Feature (Cốt lõi)  | Chiều dài vây (mm)                        |
| `body_mass_g`       | Số thực (Float)     | Feature (Cốt lõi)  | Khối lượng cơ thể (gram)                  |
| `sex`               | Phân loại (String)  | Feature (Tùy chọn) | Giới tính (male, female)                  |
| `year`              | Số nguyên (Integer) | Metadata           | Năm thu thập dữ liệu (2007, 2008, 2009)   |

## 3. Báo cáo chất lượng và Quy trình làm sạch

Dữ liệu gốc tuy có chất lượng tốt nhưng vẫn tồn tại một số giá trị khuyết (missing values). Các thao tác làm sạch sau đây đã được thực hiện ở file `src/data.py`:

- **Quy mô ban đầu:** 344 cá thể.
- **Ngoại lệ & Trùng lặp:** Không phát hiện hàng trùng lặp hoàn toàn.
- **Xử lý dữ liệu khuyết:**
  - Có đúng **02 cá thể** (chiếm 0.58%) bị khuyết hoàn toàn cả 4 số đo hình thái học (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`).
  - **Quyết định:** Loại bỏ hoàn toàn 2 dòng này. Lý do: Đây là bài toán dự đoán dựa trên khoảng cách hình thái học (K-NN), việc 1 cá thể mất cả 4 số đo cốt lõi khiến thuật toán không có bất kỳ cơ sở toán học nào để điền khuyết (Imputation) một cách hợp lý.
  - Các dòng bị khuyết cột `sex` (giới tính) vẫn được giữ lại vì tính năng này không mang tính sống còn đối với mô hình Baseline và Thí nghiệm 1, 2. Nếu sử dụng ở Thí nghiệm 3, cột `sex` sẽ được điền khuyết bằng mode (giá trị phổ biến nhất) thông qua Pipeline trên tập Train.
- **Quy mô sau làm sạch:** 342 cá thể.

## 4. Giao thức chia tập (Data Split)

Dữ liệu 342 cá thể được chia ngẫu nhiên và phân tầng (`stratify=y`) nhằm bảo toàn tỷ lệ 3 loài, với hạt giống ngẫu nhiên `random_state=42`:

- **Tập Huấn luyện (Train):** 70% (239 dòng) - Dùng để học siêu tham số, thiết lập StandardScaler.
- **Tập Xác thực (Validation):** 15% (51 dòng) - Dùng để đánh giá đối chứng các Thí nghiệm.
- **Tập Kiểm thử (Test):** 15% (52 dòng) - Đóng băng hoàn toàn, chỉ chạm vào 1 lần duy nhất ở cuối quy trình để báo cáo hiệu năng thực tế.
