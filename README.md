# 🌍 Satellite Image Segmentation with U-Net & ResNet-50

## 📝 Giới thiệu

Dự án ứng dụng Deep Learning (Thị giác máy tính) để tự động nhận diện, phân loại và trích xuất ranh giới địa hình từ hình ảnh vệ tinh. Mô hình được thiết kế để giải quyết bài toán Semantic Segmentation (Phân đoạn ngữ nghĩa) ở cấp độ điểm ảnh (pixel-level), hỗ trợ ứng dụng trong quy hoạch đô thị và giám sát tài nguyên môi trường.

Hệ thống hướng tới triển khai thực tế trên nền tảng Flycam/Drone thời gian thực kết hợp Ground Control Station (GCS), phục vụ cho các bài toán giám sát môi trường, quản lý tài nguyên và quan trắc địa hình tự động.

---

# 🚀 Công nghệ và Nền tảng

* **Ngôn ngữ:** Python 3.12
* **Framework:** PyTorch (với thư viện `segmentation-models-pytorch`)
* **Môi trường huấn luyện:** Kaggle Notebooks (GPU Tesla T4/P100)
* **Kiến trúc triển khai cuối cùng:** **U-Net + ResNet-50 + Hybrid Dilated Convolution (HDC)**
* **Kiến trúc R&D đã thử nghiệm:** Vanilla U-Net, U-Net + ResNet-50, Hybrid CNN-Transformer
* **Thư viện bổ trợ:** Albumentations, OpenCV, NumPy, Tqdm, Gc

---

# 🗂️ Tập dữ liệu & Chiến lược Mapping (Datasets)

Dự án kết hợp hai bộ dữ liệu lớn để tối ưu khả năng nhận diện trên nhiều đặc điểm địa lý khác nhau:

## 1. DeepGlobe Land Cover

Ảnh vệ tinh khu vực Âu Mỹ, nhãn định dạng **RGB (3 kênh màu)**.

* Link tải dữ liệu:
  [https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)

## 2. LoveDA

Ảnh vệ tinh siêu đô thị tại Trung Quốc (Vũ Hán...), nhãn định dạng **Grayscale (1 kênh xám)**.

* Link tải dữ liệu:
  [https://www.kaggle.com/datasets/mohammedjaveed/loveda-dataset](https://www.kaggle.com/datasets/mohammedjaveed/loveda-dataset)

---

# 🎯 Bảng Mapping nhãn đồng nhất (Label Encoding)

Hệ thống sử dụng kỹ thuật xử lý ảnh để đồng nhất các giá trị màu và pixel về 7 lớp đối tượng chuẩn:

| STT | Lớp đối tượng (Class)     | Màu DeepGlobe (RGB) | Giá trị LoveDA (Pixel) |
| :-: | :------------------------ | :-----------------: | :--------------------: |
|  0  | **Background / Unknown**  |      (0, 0, 0)      |            0           |
|  1  | **Urban (Building/Road)** |    (0, 255, 255)    |          1, 2          |
|  2  | **Agriculture**           |    (255, 255, 0)    |            6           |
|  3  | **Forest**                |     (0, 255, 0)     |            5           |
|  4  | **Water**                 |     (0, 0, 255)     |            3           |
|  5  | **Barren**                |   (255, 255, 255)   |            4           |
|  6  | **Rangeland**             |    (255, 0, 255)    |            -           |

---

# 🧠 Kỹ thuật Huấn luyện đặc trưng (Technical Implementation)

## Smart Data Loading

Xây dựng lớp `KaggleCombinedDataset` tự động ánh xạ Mask dựa trên tiền tố file (`DG_` hoặc `LDA_`), xử lý ngoại lệ để đảm bảo luồng huấn luyện liên tục.

## Combo Loss Function

Kết hợp:

* `DiceLoss`
* `CrossEntropyLoss`

giúp tối ưu đồng thời:

* Độ chính xác pixel.
* Cấu trúc vùng phân đoạn.
* Độ sắc nét của ranh giới địa hình.

## Data Augmentation

Sử dụng Albumentations:

* `HorizontalFlip`
* `VerticalFlip`
* `RandomRotate90`
* `Normalize ImageNet`

để tăng khả năng tổng quát hóa cho mô hình.

## Tối ưu phần cứng

Quản lý bộ nhớ GPU hiệu quả thông qua:

* `gc.collect()`
* `torch.cuda.empty_cache()`
* `pin_memory=True`

nhằm tránh lỗi Out-of-Memory (OOM) trên Kaggle.

---

# 📂 Cấu trúc mã nguồn

```text
.
├── notebooke683bcc9b9.ipynb    # File chạy chính trên Kaggle (Data -> Train -> Eval)
└── unet_resnet50_V3_Master.pth # File trọng số mô hình sau khi huấn luyện
```

---

# 📊 Minh họa Kết quả (Results)

|                                        Ảnh Vệ tinh Gốc                                       |                               Kết quả Dự đoán (Prediction)                               |
| :------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| ![Original](https://github.com/user-attachments/assets/10e947d0-07f2-4d80-b0d0-af5b276952c9) | ![Pred](https://github.com/user-attachments/assets/a119804f-810d-4280-8e32-b1cda520d1e1) |

---

# 📈 Quá trình R&D và Tiến hóa Kiến trúc

Trong quá trình nghiên cứu, hệ thống đã trải qua nhiều thế hệ kiến trúc khác nhau nhằm cân bằng giữa:

* Độ chính xác phân đoạn.
* Tốc độ suy luận thời gian thực.
* Chi phí tài nguyên GPU/VRAM.
* Khả năng triển khai Edge AI.

## Bảng tiến hóa mô hình

| Phiên bản   | Kiến trúc               | Đặc điểm chính                                            | Trạng thái                 |
| :---------- | :---------------------- | :-------------------------------------------------------- | :------------------------- |
| **V1 & V2** | Vanilla U-Net           | Thiết lập baseline Semantic Segmentation ban đầu          | Đã loại bỏ                 |
| **V3**      | U-Net + ResNet-50       | Tăng khả năng trích xuất đặc trưng bằng Transfer Learning | Bản ổn định                |
| **V4**      | U-Net + ResNet-50 + HDC | Mở rộng Receptive Field, tối ưu Edge AI                   | Phiên bản triển khai chính |
| **V5**      | Hybrid CNN-Transformer  | Thử nghiệm Self-Attention toàn cục                        | Thất bại trong thực tế     |

---

## V1 & V2 — Vanilla U-Net

### Kiến trúc

Mạng U-Net nguyên bản với Encoder–Decoder đối xứng và Skip Connection.

### Hạn chế

* Khả năng học đặc trưng sâu yếu.
* Biên phân đoạn mờ.
* Vỡ hạt mạnh.
* mIoU thấp.
* Không phù hợp triển khai thực tế.

---

## V3 — U-Net + ResNet-50

### Điểm nâng cấp

* ResNet-50 pretrained ImageNet.
* Residual Block.
* Transfer Learning.

### Cải thiện

* Giảm Vanishing Gradient.
* Tăng tốc độ hội tụ.
* Trích xuất đặc trưng mạnh hơn.
* Nhận diện Urban và Vegetation tốt hơn.

### Hạn chế

* Receptive Field còn hạn chế.
* Chưa xử lý tốt vật thể đa tỷ lệ.

---

## V4 — U-Net + ResNet-50 + HDC (Phiên bản tối ưu)

### Điểm nâng cấp

Bổ sung mô-đun Hybrid Dilated Convolution (HDC) tại Bottleneck của U-Net.

### Cải thiện đạt được

* Mở rộng Receptive Field.
* Giảm Gridding Effect.
* Giữ chi phí tính toán thấp.
* Tối ưu Real-time Inference.

### Hiệu năng

| Metric                |   Giá trị  |
| :-------------------- | :--------: |
| **mIoU**              | **54.36%** |
| Độ phức tạp tính toán |   `O(N)`   |
| Tối ưu Edge AI        |     Có     |
| Real-time             |     Có     |

---

## V5 — Trans-HDC (Hybrid CNN-Transformer)

### Ý tưởng

Kết hợp:

* CNN.
* Vision Transformer.
* Self-Attention.

### Vấn đề phát sinh

* Tràn VRAM (OOM).
* FPS giảm mạnh.
* Overfitting.
* Yêu cầu dataset cực lớn.

### Nguyên nhân

Độ phức tạp Self-Attention tăng theo:

`O(N²)`

nên không phù hợp với hệ thống Edge AI thời gian thực.

---

## Kết luận kiến trúc triển khai

Mặc dù Transformer có tiềm năng tăng độ chính xác lý thuyết, nhưng:

* Chi phí tài nguyên quá lớn.
* FPS thấp.
* Không phù hợp Drone thời gian thực.

Do đó:

**U-Net + ResNet-50 + HDC**

được lựa chọn là kiến trúc triển khai cuối cùng cho hệ thống.

---

# 🎯 Ứng Dụng Thực Tiễn & Đối Tượng Sử Dụng (Target Audience)

Hệ thống được thiết kế với Dashboard trực quan cùng luồng xử lý tự động hóa nhằm hỗ trợ giám sát không gian thời gian thực cho nhiều lĩnh vực chuyên môn.

---

## 🌳 Kiểm lâm & Quản lý Lâm nghiệp

### Ứng dụng

* Phát hiện cháy rừng.
* Theo dõi suy thoái rừng.
* Giám sát khai thác gỗ trái phép.
* Phân tích diện tích Vegetation và Barren theo thời gian thực.

---

## 🏙️ Quy hoạch Đô thị & Smart City

### Ứng dụng

* Đánh giá tỷ lệ bê tông hóa.
* Theo dõi mở rộng đô thị.
* Phân tích Urban vs không gian tự nhiên.
* Nội suy diện tích bằng GSD.

---

## 💧 Quan trắc Môi trường & Thủy lợi

### Ứng dụng

* Theo dõi diện tích mặt nước.
* Giám sát hồ chứa.
* Theo dõi sạt lở ven sông.
* Quan trắc biến động môi trường.

---

## 🚁 Ứng phó Khẩn cấp & Cứu nạn

### Ứng dụng

* Phân đoạn vùng ngập lụt.
* Xác định vùng an toàn.
* Hỗ trợ điều phối cứu hộ.
* Tăng tốc phản ứng hiện trường.

---

# 🚧 Hạn chế & Hướng phát triển (Future Roadmap)

## Hạn chế hiện tại

Mô hình hiện xử lý tốt trên các vùng diện tích vừa phải. Đối với ảnh vệ tinh diện tích lớn cấp độ Thành phố với độ phân giải cực cao, hệ thống vẫn gặp khó khăn về:

* Tài nguyên tính toán.
* VRAM.
* Độ chi tiết biên phân đoạn.

Ngoài ra, các kiến trúc Transformer vẫn chưa phù hợp với Edge AI thời gian thực.

---

## Lộ trình tương lai

### 1. Change Detection

So sánh biến động diện tích theo thời gian để:

* Theo dõi đô thị hóa.
* Giám sát khai thác rừng.
* Quan trắc môi trường tự động.

### 2. Web Monitoring System

Phát triển Web Dashboard cho phép:

* Upload ảnh vệ tinh.
* Chạy AI trực tuyến.
* Phân tích diện tích tự động.

### 3. Large-scale Inference

Áp dụng:

* Sliding Window Inference.
* Tile-based Segmentation.

để xử lý ảnh diện tích lớn.

### 4. Adaptive Multi-scale Segmentation

Nghiên cứu xử lý đa tỷ lệ theo:

* Độ cao Flycam.
* Góc camera.
* GSD thực tế.

---

# 🛠 Hướng dẫn trải nghiệm

1. Tải file `.ipynb` lên Kaggle.
2. Thêm dataset `DeepGlobe` và `LoveDA`.
3. Kích hoạt GPU Tesla T4/P100.
4. Chạy tuần tự các cell để train và đánh giá mô hình.

---

# 👥 Tác giả

**Huỳnh Bá Duy**
AI/Data Engineer Intern
