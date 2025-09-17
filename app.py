import streamlit as st
import cv2
import tempfile
import numpy as np
import time
from ultralytics import YOLO
from utils import save_for_retrain, init_recorder
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# ⚠️ đặt ngay sau import
st.set_page_config(page_title="Wood Defect Detection", layout="wide")

st.title("🔍 Wood Defect Detection System")

# Load model
if "model_path" not in st.session_state:
    st.session_state["model_path"] = "models/best.pt"
model = YOLO(st.session_state["model_path"])


with st.sidebar:
    uploaded_model = st.file_uploader(
        "🔄 Chọn file model mới",
        type=["pt", "onnx", "engine", "tflite", "mlmodel", "pb", "torchscript", "ncnn"],
        key="model_file_uploader",
        help="Limit 200MB per file • PT, ONNX, ENGINE, TFLITE, MLMODEL, PB, TORCHSCRIPT, NCNN"
    )
    st.markdown(
        "<div style='color:gray;font-size:13px;margin-top:-10px;'>"
        "Drag and drop file here<br>Limit 200MB per file • PT, ONNX, ENGINE, "
        "TFLITE, MLMODEL, PB, TORCHSCRIPT, NCNN"
        "</div>",
        unsafe_allow_html=True
    )

    if uploaded_model is not None:
        import tempfile, shutil
        temp_model_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix="." + uploaded_model.name.split(".")[-1]
        ).name
        with open(temp_model_path, "wb") as f:
            shutil.copyfileobj(uploaded_model, f)
        st.session_state["model_path"] = temp_model_path
        st.success(f"✅ Đã chọn model: {uploaded_model.name}")


# Sidebar
conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25, 0.05)
source = st.sidebar.radio("Chọn nguồn dữ liệu", ["Camera", "Upload Ảnh", "Upload Video"])
record = st.sidebar.checkbox("🎥 Ghi lại quá trình detect", value=False)

frame_placeholder = st.empty()
status_placeholder = st.empty()

# --- Hàm xử lý video/camera realtime ---
def process_video(cap, record=False):
    out, path = (None, None)
    if record:
        out, path = init_recorder()
        st.sidebar.success(f"Đang ghi: {path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        results = model.predict(frame, conf=conf, imgsz=640, verbose=False)
        annotated = results[0].plot()

        # Hiển thị realtime
        frame_placeholder.image(annotated, channels="BGR", width=640)
        status_placeholder.write(f"⏱️ {time.strftime('%H:%M:%S')}")

        # Nếu có ghi hình
        if record and out is not None:
            out.write(annotated)

        time.sleep(0.03)  # khoảng 30 FPS

    cap.release()
    if out: out.release()


# --- Camera Realtime ---
if source == "Camera":
    cap = cv2.VideoCapture(0)
    process_video(cap, record=record)

# --- Upload Ảnh ---
# --- Upload Ảnh ---
elif source == "Upload Ảnh":
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        results = model.predict(img, conf=conf, imgsz=640, verbose=False)
        annotated = results[0].plot()

        st.image(annotated, channels="BGR", width=640, caption="Kết quả detect")

        # Feedback
        st.subheader("📋 Đánh giá kết quả")
        correct = st.radio("Kết quả detect có đúng không?", ["Đúng", "Sai"], horizontal=True)

        if correct == "Sai":
            st.warning("⚠️ Vui lòng chọn nhãn và vẽ nhiều bbox (mỗi bbox một nhãn, màu khác nhau)")
            classes = model.names
            class_names = list(classes.values())
            # Gán màu cho từng nhãn
            color_palette = ["#FF0000", "#00FF00", "#0000FF", "#FFA500", "#800080", "#00FFFF", "#FFC0CB", "#FFFF00"]
            color_map = {name: color_palette[i % len(color_palette)] for i, name in enumerate(class_names)}
            # Chọn nhãn hiện tại
            true_label = st.selectbox("Chọn nhãn hiện tại để vẽ bbox", options=class_names)
            # Vẽ bbox mới trên ảnh gốc
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            canvas_result = st_canvas(
                fill_color=color_map[true_label] + "80",  # alpha
                stroke_width=3,
                stroke_color=color_map[true_label],
                background_image=img_pil,
                update_streamlit=True,
                height=img.shape[0],
                width=img.shape[1],
                drawing_mode="rect",
                key="canvas_multi_bbox"
            )
            if st.button("💾 Lưu các bbox"):
                if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
                    labels = []
                    objects = canvas_result.json_data["objects"]
                    # Ánh xạ màu về nhãn
                    color_to_class = {v.lower(): k for k, v in color_map.items()}
                    for obj in objects:
                        left = obj["left"]
                        top = obj["top"]
                        width_box = obj["width"]
                        height_box = obj["height"]
                        # Convert sang YOLO
                        x_center = (left + width_box / 2) / img.shape[1]
                        y_center = (top + height_box / 2) / img.shape[0]
                        w = width_box / img.shape[1]
                        h = height_box / img.shape[0]
                        # Lấy nhãn từ màu vẽ
                        stroke = obj.get("stroke", "#FF0000").lower()
                        # Loại bỏ alpha nếu có
                        if len(stroke) > 7:
                            stroke = stroke[:7]
                        cls_name = color_to_class.get(stroke, class_names[0])
                        cls_id = class_names.index(cls_name)
                        labels.append((cls_id, x_center, y_center, w, h))
                    img_path, label_path = save_for_retrain(img, labels=labels, class_names=class_names)
                    st.success(f"✅ Đã lưu {len(labels)} bbox vào {label_path}")
                else:
                    st.error("❌ Bạn cần vẽ ít nhất một bbox trước khi lưu!")


# --- Upload Video ---
# --- Upload Video ---
elif source == "Upload Video":
    uploaded_video = st.file_uploader("Chọn video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        current_frame = None
        stop_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=conf, imgsz=640, verbose=False)
            annotated = results[0].plot()

            frame_placeholder.image(annotated, channels="BGR", width=640)
            status_placeholder.write(f"🎥 Đang xử lý video... {time.strftime('%H:%M:%S')}")
            current_frame = frame.copy()

            # Nút chụp ảnh tại frame hiện tại
            if st.button("📸 Chụp frame này"):
                stop_frame = current_frame
                break

        cap.release()

        # Nếu user chụp lại frame
        if stop_frame is not None:
            st.subheader("🖼 Frame được chụp từ video")
            results = model.predict(stop_frame, conf=conf, imgsz=640, verbose=False)
            annotated = results[0].plot()

            st.image(annotated, channels="BGR", width=640, caption="Kết quả detect")

            correct = st.radio("Kết quả detect có đúng không?", ["Đúng", "Sai"], horizontal=True)
            if correct == "Sai":
                st.warning("⚠️ Chọn nhãn đúng và vẽ lại bbox trên frame")

                classes = model.names
                true_label = st.selectbox("Chọn nhãn đúng", options=list(classes.values()))

                from streamlit_drawable_canvas import st_canvas
                from PIL import Image
                img_rgb = cv2.cvtColor(stop_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=3,
                    stroke_color="#FF0000",
                    background_image=img_pil,
                    update_streamlit=True,
                    height=stop_frame.shape[0],
                    width=stop_frame.shape[1],
                    drawing_mode="rect",
                    key="canvas_video_bbox"
                )

                if st.button("💾 Lưu frame sai nhãn"):
                    bbox = None
                    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
                        obj = canvas_result.json_data["objects"][0]
                        left = obj["left"]
                        top = obj["top"]
                        width_box = obj["width"]
                        height_box = obj["height"]

                        # Chuyển sang YOLO format
                        x_center = (left + width_box / 2) / stop_frame.shape[1]
                        y_center = (top + height_box / 2) / stop_frame.shape[0]
                        w = width_box / stop_frame.shape[1]
                        h = height_box / stop_frame.shape[0]
                        bbox = (x_center, y_center, w, h)

                    if bbox:
                        img_path, label_path = save_for_retrain(stop_frame, true_label, model.names, bbox)
                        st.success(f"✅ Đã lưu vào {img_path}, {label_path}")
                    else:
                        st.error("❌ Bạn cần vẽ một bbox trước khi lưu!")
