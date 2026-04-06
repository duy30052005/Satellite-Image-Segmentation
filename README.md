# 🌍 Satellite Image Segmentation with U-Net & ResNet-50

## 📝 Giới thiệu
Dự án ứng dụng Deep Learning (Thị giác máy tính) để tự động nhận diện, phân loại và trích xuất ranh giới địa hình từ hình ảnh vệ tinh. Mô hình được thiết kế để giải quyết bài toán Semantic Segmentation (Phân đoạn ngữ nghĩa) ở cấp độ điểm ảnh (pixel-level), hỗ trợ ứng dụng trong quy hoạch đô thị và giám sát môi trường.

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
Hệ thống sử dụng kỹ thuật xử lý ảnh để đưa các giá trị màu và pixel về các lớp đối tượng chuẩn:

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

* **Smart Data Loading:** Xây dựng lớp `KaggleCombinedDataset` tự động ánh xạ Mask dựa trên tiền tố file (`DG_` hoặc `LDA_`), xử lý ngoại lệ file lỗi để đảm bảo luồng huấn luyện liên tục.
* **Combo Loss Function:** Kết hợp `DiceLoss` (giúp biên vật thể sắc nét) và `CrossEntropyLoss` (tối ưu độ chính xác pixel).
* **Augmentation:** Sử dụng **Albumentations** (`HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `Normalize ImageNet`) để tăng tính bền vững cho mô hình.
* **Tối ưu phần cứng:** Quản lý bộ nhớ GPU hiệu quả thông qua `gc.collect()`, `torch.cuda.empty_cache()` và `pin_memory=True` để tránh lỗi Out-of-Memory (OOM).

## 📂 Cấu trúc mã nguồn
```text
.
├── notebooke683bcc9b9.ipynb    # File chạy chính trên Kaggle (Data -> Train -> Eval)
└── unet_resnet50_V3_Master.pth # File trọng số mô hình sau khi huấn luyện
```

## 📊 Minh họa Kết quả (Results)



| Ảnh Vệ tinh Gốc | Ground Truth Mask | Kết quả Dự đoán (Prediction) |
|:---:|:---:|:---:|
| ![Original](https://github.com/user-attachments/assets/link_anh_goc) | ![Truth](https://github.com/user-attachments/assets/link_anh_thuc_te) | ![Pred](https://github.com/user-attachments/assets/link_anh_du_doan) |

## 🛠 Hướng dẫn trải nghiệm
1. Tải file `.ipynb` lên môi trường Kaggle.
2. Thêm 2 bộ dữ liệu `DeepGlobe` và `LoveDA` vào phần Input.
3. Kích hoạt **GPU T4 x2** hoặc **P100**.
4. Chạy toàn bộ các cell để thực hiện huấn luyện (mặc định 40 Epochs) và kiểm tra kết quả trực quan ở cuối Notebook.

## 👥 Tác giả
* **Tâm Khang** - AI/Data Engineer Intern
