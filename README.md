# 🌍 Satellite Image Segmentation with U-Net & ResNet-50

## 📝 Giới thiệu
Dự án ứng dụng Deep Learning (Thị giác máy tính) để tự động nhận diện, phân loại và trích xuất ranh giới địa hình từ hình ảnh vệ tinh. Mô hình được thiết kế để giải quyết bài toán Semantic Segmentation (Phân đoạn ngữ nghĩa) ở cấp độ điểm ảnh (pixel-level), hỗ trợ ứng dụng trong quy hoạch đô thị và giám sát tài nguyên môi trường.

## 🚀 Công nghệ và Nền tảng
* **Ngôn ngữ:** Python 3.12
* **Framework:** PyTorch (với thư viện `segmentation-models-pytorch`)
* **Môi trường huấn luyện:** Kaggle Notebooks (GPU Tesla T4/P100)
* **Kiến trúc:** **U-Net** kết hợp **ResNet-50** Backbone (Transfer Learning)
* **Thư viện bổ trợ:** Albumentations, OpenCV, NumPy, Tqdm, Gc

## 🗂️ Tập dữ liệu & Chiến lược Mapping (Datasets)
Dự án kết hợp hai bộ dữ liệu lớn để tối ưu khả năng nhận diện trên nhiều đặc điểm địa lý khác nhau:

1. **DeepGlobe Land Cover:** Ảnh vệ tinh khu vực Âu Mỹ, nhãn định dạng **RGB (3 kênh màu)**.
   * [Link tải dữ liệu](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)
2. **LoveDA:** Ảnh vệ tinh siêu đô thị tại Trung Quốc (Vũ Hán...), nhãn định dạng **Grayscale (1 kênh xám)**.
   * [Link tải dữ liệu](https://www.kaggle.com/datasets/mohammedjaveed/loveda-dataset)

### 🎯 Bảng Mapping nhãn đồng nhất (Label Encoding)
Hệ thống sử dụng kỹ thuật xử lý ảnh để đồng nhất các giá trị màu và pixel về 7 lớp đối tượng chuẩn:

| STT | Lớp đối tượng (Class) | Màu DeepGlobe (RGB) | Giá trị LoveDA (Pixel) |
|:---:|:---|:---:|:---:|
| 0 | **Background / Unknown** | (0, 0, 0) | 0 |
| 1 | **Urban (Building/Road)** | (0, 255, 255) | 1, 2 |
| 2 | **Agriculture** | (255, 255, 0) | 6 |
| 3 | **Forest** | (0, 255, 0) | 5 |
| 4 | **Water** | (0, 0, 255) | 3 |
| 5 | **Barren** | (255, 255, 255) | 4 |
| 6 | **Rangeland** | (255, 0, 255) | - |

## 🧠 Kỹ thuật Huấn luyện đặc trưng (Technical Implementation)

* **Smart Data Loading:** Xây dựng lớp `KaggleCombinedDataset` tự động ánh xạ Mask dựa trên tiền tố file (`DG_` hoặc `LDA_`), xử lý ngoại lệ để đảm bảo luồng huấn luyện liên tục.
* **Combo Loss Function:** Kết hợp `DiceLoss` (tối ưu cấu trúc vùng) và `CrossEntropyLoss` (tối ưu độ chính xác pixel), giúp ranh giới các vùng địa hình sắc nét hơn.
* **Augmentation:** Sử dụng **Albumentations** (`HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `Normalize ImageNet`) để tăng khả năng tổng quát hóa cho mô hình.
* **Tối ưu phần cứng:** Quản lý bộ nhớ GPU hiệu quả thông qua `gc.collect()`, `torch.cuda.empty_cache()` và `pin_memory=True` để tránh lỗi Out-of-Memory (OOM) trên Kaggle.

## 📂 Cấu trúc mã nguồn
```text
.
├── notebooke683bcc9b9.ipynb    # File chạy chính trên Kaggle (Data -> Train -> Eval)
└── unet_resnet50_V3_Master.pth # File trọng số mô hình sau khi huấn luyện
```

## 📊 Minh họa Kết quả (Results)

| Ảnh Vệ tinh Gốc | Kết quả Dự đoán (Prediction) |
|:---:|:---:|
| ![Original](https://github.com/user-attachments/assets/10e947d0-07f2-4d80-b0d0-af5b276952c9) | ![Pred](https://github.com/user-attachments/assets/a119804f-810d-4280-8e32-b1cda520d1e1) |

## 🚧 Hạn chế & Hướng phát triển (Future Roadmap)

### Hạn chế hiện tại
Mô hình hiện xử lý tốt trên các vùng diện tích vừa phải. Đối với ảnh vệ tinh vùng rộng cấp độ Quận/Thành phố có độ phân giải mặt đất cực cao, hệ thống vẫn gặp thách thức về tài nguyên tính toán và độ chi tiết của ranh giới khi bị nén ảnh.

### Lộ trình tương lai
1. **Phân tích biến động (Change Detection):** Xây dựng thuật toán so sánh số lượng Pixel giữa các thời điểm để tính toán tốc độ đô thị hóa hoặc tốc độ khai thác rừng tự động.
2. **Web Monitoring System:** Phát triển giao diện Web tích hợp mô hình để người dùng upload ảnh vệ tinh và nhận kết quả phân tích diện tích địa hình trực tuyến.
3. **Large-scale Inference:** Áp dụng kỹ thuật **Sliding Window Inference** để xử lý ảnh diện tích lớn (cấp độ thành phố) mà không làm giảm chất lượng nhận diện.

## 🛠 Hướng dẫn trải nghiệm
1. Tải file `.ipynb` lên môi trường Kaggle.
2. Thêm 2 bộ dữ liệu `DeepGlobe` và `LoveDA` vào phần Input.
3. Kích hoạt **GPU T4 x2** hoặc **P100**.
4. Chạy tuần tự các cell để thực hiện huấn luyện và kiểm tra kết quả trực quan ở cuối Notebook.

## 👥 Tác giả
* **Huỳnh Bá Duy** - AI/Data Engineer Intern
