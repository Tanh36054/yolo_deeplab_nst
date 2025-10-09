import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models

# ------------------- Load Models -------------------
# YOLOv8
yolo_model = YOLO("yolov8n.pt")

# DeepLabv3 (PyTorch pretrained)
deeplab_model = models.segmentation.deeplabv3_resnet50(pretrained=True).eval()

# ------------------- DeepLab Function -------------------
def run_deeplab(image_np):
    h, w, _ = image_np.shape
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(520),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    inp = transform(image_np).unsqueeze(0)

    with torch.no_grad():
        out = deeplab_model(inp)['out'][0]
    seg_map = out.argmax(0).byte().cpu().numpy()
    
    # resize seg_map về kích thước ban đầu
    seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Tạo mask màu ngẫu nhiên
    mask = np.zeros_like(image_np)
    for label in np.unique(seg_map):
        mask[seg_map == label] = np.random.randint(0, 255, size=3)

    overlay = cv2.addWeighted(image_np, 0.6, mask, 0.4, 0)
    return overlay

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="YOLO + DeepLab Demo", layout="wide")
st.title("🚀 YOLOv8 + DeepLab Demo")
st.header("Group 08")

st.markdown(
    """
    <p style='font-size:16px; font-weight:normal;'>
    Ngô Thị Thanh Vân – Nguyễn Thế Anh – Nguyễn Tiến Hưng – Quang Hồng Ánh Sứ
    </p>
    """,
    unsafe_allow_html=True
)

# Sidebar chọn chế độ
task = st.sidebar.radio("Chọn mô hình:", ["YOLOv8", "DeepLab"])
mode = st.sidebar.radio("Chọn chế độ:", ["Ảnh", "Video", "Webcam"])

# ------------------- YOLOv8 -------------------
if task == "YOLOv8":

    # ẢNH
    if mode == "Ảnh":
        uploaded_file = st.file_uploader("Tải ảnh", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            import os
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            results = yolo_model.predict(source=tmp_path, conf=0.5)
            for r in results:
                img = r.plot()
                st.image(img, caption="Kết quả YOLOv8", channels="BGR", use_container_width=True)

    # VIDEO
    elif mode == "Video":
        uploaded_file = st.file_uploader("Tải video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            import os
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            st.video(tmp_path)
            st.write("👉 Kết quả.")
            
            cap = cv2.VideoCapture(tmp_path)
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = yolo_model.predict(frame, conf=0.5)
                annotated_frame = results[0].plot()
                stframe.image(annotated_frame, channels="BGR", use_container_width=True)
            cap.release()

    # WEBCAM
    elif mode == "Webcam":
        st.write("👉 Nhấn Start để bật webcam nhận diện YOLOv8.")
        if st.button("Start"):
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = yolo_model.predict(frame, conf=0.5)
                annotated_frame = results[0].plot()
                stframe.image(annotated_frame, channels="BGR", use_container_width=True)
            cap.release()

# ------------------- DeepLab -------------------
elif task == "DeepLab":
    if mode == "Ảnh":
        uploaded_file = st.file_uploader("Tải ảnh để phân đoạn", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            import os
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            img = cv2.imread(tmp_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            output = run_deeplab(img_rgb)
            st.image(output, caption="Kết quả DeepLab", channels="RGB", use_container_width=True)
    else:
        st.warning("⚠️ DeepLab chỉ hỗ trợ xử lý Ảnh trong demo này.")
