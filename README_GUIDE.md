# 🧠 YOLO + DeepLab + Neural Style Transfer Demo

Dự án này kết hợp ba mô hình xử lý ảnh:
- **YOLOv8**: phát hiện đối tượng trong ảnh/video.
- **DeepLabv3**: phân đoạn ảnh (semantic segmentation).
- **NST (Neural Style Transfer)**: biến đổi phong cách ảnh.

---

### 🚀 Cách chạy dự án

## Clone project về:
git clone https://github.com/Tanh36054/yolo_deeplab_nst.git

## Bước 1️⃣: Cài Python
Tải và cài Python phiên bản 3.10 hoặc 3.11 từ
👉 https://www.python.org/downloads/
Khi cài nhớ tick ✅ "Add Python to PATH"

## Bước 2️⃣: Tạo môi trường ảo
- Mở PowerShell hoặc CMD tại thư mục dự án, gõ:
python -m venv yolo_env
- Kích hoạt môi trường ảo:
Windows (PowerShell):
yolo_env\Scripts\activate
Linux/macOS:
source yolo_env/bin/activate
## Bước 3️⃣: Cài đặt thư viện
Trong khi môi trường ảo đang bật, gõ:
pip install ultralytics streamlit opencv-python tensorflow pillow numpy

🧠 4. Cách chạy web
✅ Chạy web YOLO + DeepLab
streamlit run app.py

🎨 Chạy web Neural Style Transfer (NST):
streamlit run nst_app.py
Chọn:
Ảnh gốc (Content Image)
Ảnh phong cách (Style Image)
Sau đó bấm Run NST để xem kết quả.


