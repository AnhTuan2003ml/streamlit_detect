import cv2
import os
import numpy as np
import time

def save_for_retrain(img, labels, class_names, save_dir="retrain_data"):
    import os, cv2, uuid
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)
    img_id = str(uuid.uuid4())
    img_path = os.path.join(save_dir, "images", f"{img_id}.jpg")
    label_path = os.path.join(save_dir, "labels", f"{img_id}.txt")
    cv2.imwrite(img_path, img)
    with open(label_path, "w") as f:
        for cls_id, x, y, w, h in labels:
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    return img_path, label_path



def init_recorders():
    import os, time, cv2
    os.makedirs("recordings/raw", exist_ok=True)
    os.makedirs("recordings/detect", exist_ok=True)
    session_name = time.strftime("%Y%m%d_%H%M%S")
    path_raw = f"recordings/raw/{session_name}.avi"
    path_detect = f"recordings/detect/{session_name}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_raw = cv2.VideoWriter(path_raw, fourcc, 20.0, (640, 480))
    out_detect = cv2.VideoWriter(path_detect, fourcc, 20.0, (640, 480))
    return out_raw, path_raw, out_detect, path_detect