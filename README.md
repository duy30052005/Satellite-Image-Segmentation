# 🌍 Satellite Image Segmentation with U-Net & ResNet-50

## 📝 Giới thiệu
Dự án ứng dụng Deep Learning (Thị giác máy tính) để tự động nhận diện, phân loại và trích xuất ranh giới địa hình từ hình ảnh vệ tinh. Mô hình được thiết kế để giải quyết bài toán Semantic Segmentation (Phân đoạn ngữ nghĩa) ở cấp độ điểm ảnh (pixel-level), hỗ trợ ứng dụng trong quy hoạch đô thị, giám sát môi trường và nông nghiệp.

## 🚀 Công nghệ và Nền tảng
* **Ngôn ngữ:** Python
* **Môi trường huấn luyện:** Kaggle Notebooks (Tận dụng sức mạnh xử lý song song của GPU).
* **Kiến trúc Mô hình:** **U-Net** kết hợp với **ResNet-50** Backbone.
* **Thư viện chính:** PyTorch / TensorFlow (Keras), OpenCV, NumPy, Matplotlib, Scikit-learn.

## 🗂️ Tập dữ liệu (Datasets)
Để đảm bảo mô hình có khả năng tổng quát hóa cao (Generalization) trên nhiều loại địa hình và mật độ dân cư khác nhau, dự án sử dụng kết hợp hai bộ dữ liệu vệ tinh lớn:

1. **DeepGlobe Land Cover Classification:**
   * **Đặc điểm:** Bao gồm hình ảnh vệ tinh tại các khu vực Châu Âu và Châu Mỹ. 
   * **Thế mạnh:** Cực kỳ phù hợp để nhận diện các vùng nông thôn, ngoại ô, hoặc khu vực đô thị có mật độ thấp xen lẫn với rừng cây và thảm thực vật.
   * **Định dạng nhãn (Mask):** Được đánh dấu chi tiết bằng kênh màu đa sắc (RGB).
   * **Link tải dữ liệu:** [Kaggle - DeepGlobe Land Cover]([link_dataset_deepglobe_cua_ban](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset))

2. **LoveDA (Land-Cover Domain Adaptive):**
   * **Đặc điểm:** Tập trung vào các hình ảnh vệ tinh tại các siêu đô thị lớn của Trung Quốc (như Vũ Hán...).
   * **Thế mạnh:** Tối ưu hóa khả năng nhận diện của mô hình tại các khu vực có mật độ đô thị hóa cực cao, nhà cửa san sát và cấu trúc giao thông phức tạp.
   * **Định dạng nhãn (Mask):** Được đánh dấu bằng kênh xám đơn sắc (1-channel Grayscale).
   * **Link tải dữ liệu:** [Kaggle - LoveDA Dataset]([link_dataset_loveda_cua_ban](https://www.kaggle.com/datasets/mohammedjaveed/loveda-dataset))

*(Lưu ý: Quá trình tiền xử lý đã được tùy chỉnh để đồng nhất hóa hai loại định dạng Mask (RGB và Grayscale) vào chung một pipeline huấn luyện).*

## 🧠 Kiến trúc Mô hình (Model Architecture)
Thay vì sử dụng U-Net truyền thống, dự án áp dụng kỹ thuật Transfer Learning để tối ưu hóa khả năng trích xuất đặc trưng:
* **Encoder (Bộ mã hóa - ResNet-50):** Sử dụng các trọng số (weights) đã được huấn luyện trước (pre-trained) của mạng ResNet-50. Khối kiến trúc này đóng vai trò trích xuất các đặc trưng không gian phức tạp từ ảnh vệ tinh mà không gặp hiện tượng suy biến đạo hàm (vanishing gradient).
* **Decoder (Bộ giải mã - U-Net):** Phục hồi lại độ phân giải ban đầu của ảnh, sử dụng các kết nối tắt (Skip Connections) từ Encoder sang Decoder để giữ lại các thông tin vị trí chi tiết, giúp ranh giới các vật thể được cắt biên sắc nét và chính xác hơn.

## 💡 Luồng xử lý dữ liệu (Workflow)

### 1. Tiền xử lý dữ liệu (Data Preprocessing & Augmentation)
* Đọc, giải mã và đồng nhất định dạng cho hai tập dữ liệu DeepGlobe và LoveDA.
* Áp dụng các kỹ thuật Data Augmentation (Xoay, lật, thay đổi độ sáng/độ tương phản, cắt ngẫu nhiên) để làm phong phú dữ liệu, giúp mô hình tăng cường độ mạnh mẽ và tránh Overfitting.

### 2. Huấn luyện (Training)
* Tận dụng bộ tăng tốc phần cứng (GPU) trên Kaggle để xử lý hàng nghìn epoch cho tập dữ liệu hình ảnh lớn.
* Lựa chọn hàm mất mát (Loss Function) phù hợp cho bài toán phân đoạn nhiều lớp.

### 3. Đánh giá (Evaluation Metrics)
Mô hình được đánh giá qua các thang đo chuẩn trong bài toán Segmentation:
* **IoU (Intersection over Union) / Jaccard Index:** Đánh giá độ trùng khớp giữa vùng dự đoán và thực tế.
* **Dice Coefficient (F1-Score):** Đo lường độ tương đồng của tập hợp pixel.
* **Pixel Accuracy:** Tỷ lệ phần trăm các điểm ảnh được phân loại đúng.

## 📂 Cấu trúc Repository
* `notebooke683bcc9b9.ipynb`: Kaggle Notebook chứa toàn bộ mã nguồn từ khâu tải dữ liệu, xây dựng Custom DataLoader xử lý RGB/Grayscale Mask, định nghĩa kiến trúc U-Net + ResNet-50, vòng lặp huấn luyện (Training loop) và trực quan hóa kết quả dự đoán.

## 📊 Minh họa Kết quả (Results)

| Ảnh Vệ tinh Gốc | Ground Truth Mask | Kết quả Dự đoán (Prediction) |
|:---:|:---:|:---:|
| ![Original](link_anh_goc_cua_ban) | ![Truth](link_anh_thuc_te_cua_ban) | ![Pred](link_anh_du_doan_cua_ban) |

---

## 🛠 Hướng dẫn trải nghiệm

Vì mô hình được huấn luyện trên môi trường Kaggle với dữ liệu lớn, cách tốt nhất để xem và chạy lại dự án là thông qua Jupyter Notebook:
1. Clone repository này về máy hoặc tải trực tiếp file `.ipynb`.
2. Tải file `.ipynb` lên Kaggle hoặc Google Colab.
3. Tải hai bộ dữ liệu `DeepGlobe Land Cover` và `LoveDA` thông qua các link đính kèm ở trên và import vào môi trường.
4. Bật chế độ tăng tốc GPU (Runtime > Change runtime type > GPU).
5. Chạy tuần tự các Cell để xem quá trình huấn luyện và kết quả trực quan ở cuối file.

## 👥 Tác giả
* **Tâm Khang** - AI/Data Engineer Intern
