import streamlit as st
import cv2
import tempfile
import numpy as np
import time
from ultralytics import YOLO
from utils import save_for_retrain, init_recorders
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from streamlit_autorefresh import st_autorefresh

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

if "cameras" not in st.session_state:
    st.session_state["cameras"] = []
if "add_cam_mode" not in st.session_state:
    st.session_state["add_cam_mode"] = False
if "new_cam_type" not in st.session_state:
    st.session_state["new_cam_type"] = "USB"
if "new_cam_source" not in st.session_state:
    st.session_state["new_cam_source"] = 0

source = st.sidebar.radio("Chọn nguồn dữ liệu", ["Camera", "Upload Ảnh", "Upload Video"])
# Camera Section - Refactored
# Camera Section - Fixed Version
if source == "Camera":
    st.header("📷 Camera Detection")
    
    # Initialize session states
    if "cameras" not in st.session_state:
        st.session_state["cameras"] = {}
    if "show_add_camera" not in st.session_state:
        st.session_state["show_add_camera"] = False
    if "recording_states" not in st.session_state:
        st.session_state["recording_states"] = {}
    if "camera_running" not in st.session_state:
        st.session_state["camera_running"] = {}
    
    # ✅ Auto refresh nhanh hơn cho realtime
    st_autorefresh(interval=100, key="camera_refresh")  # 100ms thay vì 1000ms
    
    # Nút thêm camera
    if st.button("➕ Thêm Camera", type="primary"):
        st.session_state["show_add_camera"] = True
    
    # Form thêm camera
    if st.session_state["show_add_camera"]:
        with st.form("add_camera_form"):
            st.write("### 🎥 Cấu hình Camera mới")
            
            camera_name = st.text_input("Tên camera:", value=f"Camera {len(st.session_state['cameras'])+1}")
            camera_type = st.radio("Loại camera:", ["USB", "IP"], horizontal=True)
            
            if camera_type == "USB":
                camera_source = st.selectbox("USB Camera:", [0, 1, 2, 3, 4])
            else:
                camera_source = st.text_input("URL IP Camera:", 
                                            placeholder="rtsp://admin:password@192.168.1.100:554/stream")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("✅ Xác nhận", type="primary"):
                    # Test camera connection
                    with st.spinner("Đang test kết nối..."):
                        try:
                            test_cap = cv2.VideoCapture(camera_source)
                            if test_cap.isOpened():
                                ret, frame = test_cap.read()
                                test_cap.release()
                                
                                if ret:
                                    # Tạo camera mới
                                    camera_id = f"cam_{int(time.time())}"
                                    st.session_state["cameras"][camera_id] = {
                                        "name": camera_name,
                                        "type": camera_type,
                                        "source": camera_source,
                                        "cap": None,
                                        "status": "ready"  # ✅ Đổi từ "stopped" thành "ready"
                                    }
                                    st.session_state["recording_states"][camera_id] = {
                                        "is_recording": False,
                                        "video_writer": None,
                                        "start_time": None
                                    }
                                    st.session_state["camera_running"][camera_id] = True  # ✅ Auto start
                                    
                                    st.success(f"✅ Camera {camera_name} đã được thêm và sẽ bắt đầu ngay!")
                                    st.session_state["show_add_camera"] = False
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("❌ Camera kết nối được nhưng không có tín hiệu!")
                            else:
                                st.error(f"❌ Không thể kết nối camera: {camera_source}")
                        except Exception as e:
                            st.error(f"❌ Lỗi: {str(e)}")
            
            with col2:
                if st.form_submit_button("❌ Hủy"):
                    st.session_state["show_add_camera"] = False
                    st.rerun()
    
    # ✅ Hiển thị cameras với logic đơn giản hóa
    if st.session_state["cameras"]:
        st.markdown("---")
        
        # Grid layout cho cameras
        camera_ids = list(st.session_state["cameras"].keys())
        num_cols = min(len(camera_ids), 2)  # Tối đa 2 cột
        
        if num_cols > 0:
            cols = st.columns(num_cols)
            
            for idx, camera_id in enumerate(camera_ids):
                camera = st.session_state["cameras"][camera_id]
                col_idx = idx % num_cols
                
                with cols[col_idx]:
                    # Camera container
                    with st.container():
                        st.markdown(f"### 🎦 {camera['name']}")
                        
                        # Video frame placeholder
                        frame_placeholder = st.empty()
                        status_placeholder = st.empty()
                        
                        # ✅ Khởi tạo camera một lần và giữ kết nối
                        camera_should_run = st.session_state["camera_running"].get(camera_id, False)
                        
                        if camera_should_run:
                            # Khởi tạo camera nếu chưa có
                            if camera["cap"] is None:
                                try:
                                    cap = cv2.VideoCapture(camera["source"])
                                    if cap.isOpened():
                                        # Tối ưu settings
                                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                        cap.set(cv2.CAP_PROP_FPS, 30)
                                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                        
                                        camera["cap"] = cap
                                        camera["status"] = "running"
                                    else:
                                        camera["status"] = "error"
                                        camera_should_run = False
                                        st.session_state["camera_running"][camera_id] = False
                                        status_placeholder.error("❌ Không thể khởi tạo camera")
                                except Exception as e:
                                    camera["status"] = "error"
                                    camera_should_run = False
                                    st.session_state["camera_running"][camera_id] = False
                                    status_placeholder.error(f"❌ Lỗi: {str(e)}")
                            
                            # ✅ Đọc và hiển thị frame liên tục
                            if camera["status"] == "running" and camera["cap"] is not None:
                                try:
                                    cap = camera["cap"]
                                    ret, frame = cap.read()
                                    
                                    if ret:
                                        # YOLO detection
                                        results = model.predict(
                                            frame, 
                                            conf=conf, 
                                            imgsz=640,
                                            verbose=False,
                                            device='cpu'
                                        )
                                        annotated = results[0].plot()
                                        
                                        # Hiển thị frame
                                        frame_placeholder.image(annotated, channels="BGR", use_column_width=True)
                                        
                                        # Recording logic
                                        recording_state = st.session_state["recording_states"][camera_id]
                                        if recording_state["is_recording"]:
                                            if recording_state["video_writer"] is not None:
                                                recording_state["video_writer"].write(annotated)
                                            
                                            # Hiển thị trạng thái recording
                                            elapsed = int(time.time() - recording_state["start_time"])
                                            status_placeholder.success(f"🔴 RECORDING - {elapsed}s - {time.strftime('%H:%M:%S')}")
                                        else:
                                            status_placeholder.success(f"🟢 LIVE - {time.strftime('%H:%M:%S')}")
                                    
                                    else:
                                        frame_placeholder.warning("⚠️ Không đọc được frame")
                                        status_placeholder.warning("No Signal")
                                        
                                except Exception as e:
                                    frame_placeholder.error(f"❌ Lỗi xử lý: {str(e)}")
                                    status_placeholder.error("Processing Error")
                            
                        else:
                            # Camera stopped
                            frame_placeholder.info("📷 Camera đã dừng. Nhấn 'Start' để bắt đầu.")
                            status_placeholder.info("Camera Stopped")
                        
                        # ✅ Control buttons đơn giản hóa
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            # Start/Stop camera
                            if not camera_should_run:
                                if st.button("▶️ Start", key=f"start_{camera_id}"):
                                    st.session_state["camera_running"][camera_id] = True
                                    st.rerun()
                            else:
                                if st.button("⏸️ Stop", key=f"stop_{camera_id}"):
                                    # Stop camera
                                    if camera["cap"] is not None:
                                        camera["cap"].release()
                                        camera["cap"] = None
                                    
                                    # Stop recording nếu đang record
                                    recording_state = st.session_state["recording_states"][camera_id]
                                    if recording_state["is_recording"]:
                                        if recording_state["video_writer"] is not None:
                                            recording_state["video_writer"].release()
                                        recording_state["is_recording"] = False
                                        recording_state["video_writer"] = None
                                    
                                    st.session_state["camera_running"][camera_id] = False
                                    camera["status"] = "stopped"
                                    st.rerun()
                        
                        with col2:
                            # Record button
                            recording_state = st.session_state["recording_states"][camera_id]
                            if camera_should_run and not recording_state["is_recording"]:
                                if st.button("🔴 Record", key=f"record_{camera_id}"):
                                    try:
                                        # Tạo video writer
                                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                                        output_path = f"recordings/{camera['name']}_{timestamp}.mp4"
                                        
                                        # Tạo thư mục recordings nếu chưa có
                                        import os
                                        os.makedirs("recordings", exist_ok=True)
                                        
                                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                        video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
                                        
                                        recording_state["is_recording"] = True
                                        recording_state["video_writer"] = video_writer
                                        recording_state["start_time"] = time.time()
                                        recording_state["output_path"] = output_path
                                        
                                        st.success(f"🔴 Bắt đầu record: {output_path}")
                                        time.sleep(0.5)
                                        st.rerun()
                                        
                                    except Exception as e:
                                        st.error(f"❌ Lỗi record: {str(e)}")
                            elif recording_state["is_recording"]:
                                if st.button("⏹️ Stop Rec", key=f"stop_record_{camera_id}"):
                                    try:
                                        if recording_state["video_writer"] is not None:
                                            recording_state["video_writer"].release()
                                        
                                        recording_state["is_recording"] = False
                                        recording_state["video_writer"] = None
                                        
                                        st.success(f"✅ Đã lưu: {recording_state.get('output_path', 'video')}")
                                        time.sleep(0.5)
                                        st.rerun()
                                        
                                    except Exception as e:
                                        st.error(f"❌ Lỗi dừng record: {str(e)}")
                        
                        with col3:
                            # Restart camera
                            if st.button("🔄 Restart", key=f"restart_{camera_id}"):
                                try:
                                    # Release old camera
                                    if camera["cap"] is not None:
                                        camera["cap"].release()
                                        camera["cap"] = None
                                    
                                    # Stop recording nếu đang record
                                    if recording_state["is_recording"]:
                                        if recording_state["video_writer"] is not None:
                                            recording_state["video_writer"].release()
                                        recording_state["is_recording"] = False
                                        recording_state["video_writer"] = None
                                    
                                    camera["status"] = "ready"
                                    st.session_state["camera_running"][camera_id] = True  # Auto start sau restart
                                    st.success("🔄 Camera restarted!")
                                    time.sleep(0.5)
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"❌ Lỗi restart: {str(e)}")
                        
                        with col4:
                            # Delete camera
                            if st.button("🗑️ Xóa", key=f"delete_{camera_id}"):
                                try:
                                    # Cleanup camera
                                    if camera["cap"] is not None:
                                        camera["cap"].release()
                                    
                                    # Stop recording
                                    if recording_state["is_recording"]:
                                        if recording_state["video_writer"] is not None:
                                            recording_state["video_writer"].release()
                                    
                                    # Remove from session state
                                    del st.session_state["cameras"][camera_id]
                                    del st.session_state["recording_states"][camera_id]
                                    if camera_id in st.session_state["camera_running"]:
                                        del st.session_state["camera_running"][camera_id]
                                    
                                    st.success(f"✅ Đã xóa camera {camera['name']}")
                                    time.sleep(0.5)
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"❌ Lỗi xóa camera: {str(e)}")
                        
                        st.markdown("---")
    
    else:
        # No cameras message
        st.info("📷 Chưa có camera nào. Nhấn **'➕ Thêm Camera'** để bắt đầu.")
        st.markdown("""
        ### 💡 Hướng dẫn:
        - **USB Camera**: Chọn index từ 0-4 (thường là 0 hoặc 1)
        - **IP Camera**: Nhập URL đầy đủ như `rtsp://admin:123456@192.168.1.100:554/stream`
        - Sau khi thêm camera, hệ thống sẽ **tự động bắt đầu** detection realtime
        - Sử dụng các nút **Start/Stop**, **Record**, **Restart**, **Xóa** để điều khiển camera
        """)

    # Cleanup khi thoát
    def cleanup_cameras():
        """Cleanup tất cả cameras khi thoát"""
        if "cameras" in st.session_state:
            for camera_id, camera in st.session_state["cameras"].items():
                if camera.get("cap") is not None:
                    camera["cap"].release()
        
        if "recording_states" in st.session_state:
            for camera_id, recording_state in st.session_state["recording_states"].items():
                if recording_state.get("video_writer") is not None:
                    recording_state["video_writer"].release()
    
    # Register cleanup
    import atexit
    atexit.register(cleanup_cameras)

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

            # frame_placeholder.image(annotated, channels="BGR", width=640)
            # status_placeholder.write(f"🎥 Đang xử lý video... {time.strftime('%H:%M:%S')}")
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
