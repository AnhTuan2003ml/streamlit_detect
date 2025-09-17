# 1. Cài đặt ultralytics (nếu chưa có)
!pip install ultralytics

# 2. Import các thư viện cần thiết
from ultralytics import YOLO
import os

# 3. Khởi tạo model YOLOv11
# Bạn có thể chọn model size phù hợp: yolo11n.pt (nano), yolo11s.pt (small), yolo11m.pt (medium), yolo11l.pt (large), yolo11x.pt (extra large)
model = YOLO('yolo11s.pt')  # hoặc yolo11s.pt, yolo11m.pt, etc.

# 4. Đường dẫn đến file data.yaml của dataset
# Thường sẽ nằm trong thư mục dataset vừa tải về
data_path = "/content/Wood-Defect-1/data.yaml"  # Điều chỉnh path cho phù hợp

# 5. Train model
results = model.train(
    data=data_path,           # Đường dẫn đến file data.yaml
    epochs=50,               # Số epochs train (có thể điều chỉnh)
    imgsz=640,               # Kích thước ảnh input
    batch=32,                # Batch size (điều chỉnh theo GPU memory)
    device=0,                # GPU device (0 cho GPU đầu tiên, 'cpu' cho CPU)
    workers=16,               # Số worker processes
    project='/content/checkpoit',    # Thư mục lưu kết quả
    name='wood_defect_v11',  # Tên experiment
    save=True,               # Lưu checkpoint
    save_period=10,          # Lưu checkpoint mỗi 10 epochs
    val=True,                # Validate trong quá trình train
    plots=True,              # Tạo plots kết quả
    verbose=True             # In thông tin chi tiết
)

# 6. Xem kết quả training
print("Best model saved at:", results.save_dir)
print("Training metrics:", results.results_dict)

# 7. Load model đã train để test
best_model = YOLO(f'{results.save_dir}/weights/best.pt')

# 8. Test trên ảnh mới
# test_results = best_model('path/to/test/image.jpg')
# test_results[0].show()  # Hiển thị kết quả

# 9. Validate model
val_results = best_model.val()
print("Validation mAP50:", val_results.box.map50)
print("Validation mAP50-95:", val_results.box.map)

# 10. Export model sang các format khác (optional)
# best_model.export(format='onnx')  # Export sang ONNX
# best_model.export(format='tensorrt')  # Export sang TensorRT
# 10. Export model sang các format khác nhau cho từng thiết bị

# --- Jetson Nano / TX1 ---
# 1) Export ONNX (có thể convert sang TensorRT trên Jetson)
best_model.export(format="onnx", opset=12)

# 2) Export trực tiếp TensorRT (nếu chạy trên máy có TensorRT, thường phải convert ONNX -> TensorRT trên Jetson bằng trtexec)
# best_model.export(format="engine", half=True)  # half=True để FP16 tăng tốc

# --- Mac M1 Pro ---
# Export sang CoreML (.mlmodel)
best_model.export(format="coreml")

# Ngoài ra vẫn có thể chạy trực tiếp bằng PyTorch device="mps":
# results = best_model.predict(source=0, device="mps")
