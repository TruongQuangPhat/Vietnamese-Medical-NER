# Nhận dạng thực thể (NER) Y học Cổ truyền Tiếng Việt

## 1. Giới thiệu
Dự án này xây dựng hệ thống **Named Entity Recognition (NER)** chuyên sâu cho văn bản y học cổ truyền Việt Nam. Hệ thống tập trung nhận diện 6 loại thực thể chính: `DISEASE` (Bệnh), `HERB` (Dược liệu), `DOSAGE` (Liều lượng), `SYMPTOM` (Triệu chứng), `HUMAN_PART` (Bộ phận cơ thể), `PLANT_PART` (Bộ phận cây thuốc).

Dự án so sánh hai phương pháp tiếp cận:
- Baseline Method: Sử dụng nhãn cứng (Hard Labels) và hàm mất mát Cross-Entropy truyền thống.
- Proposed Method: Sử dụng nhãn mềm (Soft Labels/Probability Distribution) và hàm mất mát **KL Divergence** trên nền tảng **DeBERTa-v3** để xử lý tính nhập nhằng trong y văn.

## 2. Cấu trúc Dự án
```bash
├── data/                       # Dữ liệu sau khi đã gán nhãn thủ công
│   ├── data_new_method.jsonl   # Dữ liệu nhãn mềm (xác suất)
│   └── data_old_method.jsonl   # Dữ liệu nhãn cứng (argmax từ nhãn mềm)
├── input/                      # Chứa file DOCX thô sau OCR
│   ├── Thuc-vat-duoc.docx
│   ├── Tue-Tinh-toan-tap.docx
│   └── dieu-tri-hoc-ket-hop-y-hoc-hien-dai-va-y-hoc-co-truyen.docx
├── output/                     # Chứa file JSONL sau khi Auto-labeling
├── labels/                     # Từ điển thực thể trích xuất bởi LLMs
│   ├── list_diseases.txt
│   ├── list_herbs.txt
│   ├── list_dosages.txt
│   ├── list_human_parts.txt
│   ├── list_plant_parts.txt
│   └── list_symptoms.txt
├── src/                        # Mã nguồn xử lý dữ liệu
│   ├── auto_labeling.py        # Script gán nhãn tự động
│   └── ocr_data.py             # Script ocr data
├── notebooks/                  # Mã nguồn huấn luyện mô hình
│   ├── PretrainOldMethod.ipynb  # (Tên cũ: PretrainOldMethod.ipynb)
│   └── PretrainNewMethod.ipynb  # (Tên cũ: PretrainNewMethod.ipynb)
├── environment.yml             # Cấu hình môi trường Conda
└── README.md
```

## 3. Thiết lập Môi trường
Dự án sử dụng `conda` để quản lý môi trường. Hãy đảm bảo đã cài đặt Anaconda hoặc Miniconda.

**Bước 1: Tạo môi trường từ file `.yml`**
```bash
conda env create -f environment.yml
```

**Bước 2: Kích hoạt môi trường**
```bash
conda activate medical-ner
```

## 4. Quy trình Xử lý Dữ liệu
Quy trình bao gồm 4 giai đoạn chính để chuyển đổi từ tài liệu thô sang dữ liệu sẵn sàng huấn luyện.

**Giai đoạn 1: Trích xuất Từ điển**
- **Đầu vào:** Các file sách y học dạng `.docx` (đã OCR từ PDF).
- **Phương pháp:** Sử dụng LLMs để trích xuất các danh từ y học và phân loại vào 6 nhóm nhãn (`HERB`, `DISEASE`, `SYMPTOM`, ...).
- **Đầu ra:** Các file `.txt` lưu trong thư mục `labels/`.

**Giai đoạn 2: Gán nhãn Tự động**
Sử dụng script `auto_labeling.py` để quét từ điển và Regex (cho liều lượng) trên các file DOCX gốc. Script hỗ trợ xử lý hàng loạt (batch processing).

**Cách chạy lệnh:**
```bash
python src/auto_labeling.py --input_dir "input" --output_dir "output"
```
**Logic xử lý:**
- Đọc tất cả file `.docx` trong thư mục `input/`.
- Hợp nhất các dòng bị ngắt quãng do lỗi OCR.
- Tách câu (Sentence Segmentation).
- Gán nhãn tự động dựa trên từ điển và Regex (ưu tiên nhãn dài nhất - Longest Match).

**Kết quả:** File `.jsonl` thô được lưu vào thư mục `output/`

**Giai đoạn 3: Kiểm tra & Gán nhãn Thủ công**
- **Chọn lọc:** Lọc ra 3,000 câu chất lượng tốt nhất từ dữ liệu Auto-labeling.
- **Gán nhãn:** Các thành viên sử dụng công cụ **Doccano** để kiểm tra và sửa lỗi nhãn.
- **Hậu xử lý:**
  - `data_new_method.jsonl`: Lưu trữ dưới dạng phân phối xác suất (Soft Labels). Ví dụ: Một thực thể có thể là 80% `HERB` và 20% `PLANT`.
  - `data_old_method.jsonl`: Được suy ra từ file trên bằng cách lấy nhãn có xác suất cao nhất (Hard Labels) để làm Baseline.

**Giai đoạn 4: Huấn luyện Mô hình**

**Cấu hình chung**
- **Phần cứng:** Google Colab (GPU T4 hoặc A100).
- **Dữ liệu:** Upload thư mục data/ lên Google Drive.

**Notebook 1: PretrainOldMethod.ipynb**
- **Dữ liệu:** `data_old_method.jsonl`
- **Kiến trúc:** DeBERTa-v3
- **Hàm mất mát:** Cross-Entropy Loss.
- **Mục đích:** Tạo mô hình chuẩn để so sánh.

**Notebook 2: PretrainNewMethod.ipynb**
- **Dữ liệu:** `data_new_method.jsonl`
- **Kiến trúc:** DeBERTa-v3.
- **Hàm mất mát:** KL Divergence Loss.
- **Mục đích:** Mô hình học được độ bất định của dữ liệu, giúp cải thiện F1-score trên các thực thể nhập nhằng.

## 5. Hướng dẫn chạy trên Colab
Để chạy 2 file notebook huấn luyện, thực hiện các bước sau:
- **Chuẩn bị dữ liệu trên Drive: Tải toàn bộ cấu trúc thư mục lên Drive:**
```bash
  /content/drive/MyDrive/medical_ner/
  ├── data/
  │   ├── data_new_method.jsonl
  │   └── data_old_method.jsonl
  └── checkpoint/
  ```
- **Mở Notebook:**
  - Mở `PretrainOldMethod.ipynb` hoặc `PretrainNewMethod.ipynb`.

- **Chạy Huấn luyện:**
  - Chọn `Runtime` -> `Run all`.
  - Mô hình sẽ tự động train, đánh giá và lưu checkpoint tốt nhất vào thư mục `/checkpoint/` trên Drive.

## 6. Kết quả mong đợi
- **Artifacts:** Checkpoint mô hình (`pytorch_model.bin`, `config.json`, `tokenizer.json`) và file ánh xạ nhãn (`label_maps.json`).
- **Metrics:** Bảng báo cáo Precision, Recall, F1-score cho từng loại thực thể.
- **So sánh:** Phương pháp mới kỳ vọng đạt F1-score cao hơn và loss thấp hơn so với phương pháp cũ trên tập Test.