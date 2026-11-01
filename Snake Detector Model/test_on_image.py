import sys
import os
import cv2
import numpy as np
from ultralytics import RTDETR, YOLO
from detect_and_alarm import (
    DETECTOR_PATH,
    CLASSIFIER_PATH,
    analyze_species,
    PROXIMITY_ALERT_THRESHOLD,
    DETECTION_CONF_THRESHOLD,
    CLASSIFIER_CROP_PADDING
)

def run_pipeline_on_image(image_path, save_output=True):
    print(f"\n[INFO] Loading models...")
    detector = RTDETR(DETECTOR_PATH)
    classifier = YOLO(CLASSIFIER_PATH)

    print(f"[INFO] Processing image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not load image from '{image_path}'")
        return

    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_h * frame_w

    # 1. Run Detector
    print("[INFO] Running Stage 1 RT-DETR Snake Hunter...")
    det_results = detector(frame, conf=DETECTION_CONF_THRESHOLD, verbose=False)

    detections = []
    for r in det_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w))
            y2 = max(0, min(y2, frame_h))

            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            if bw > 10 and bh > 10:
                proximity_pct = ((bw * bh) / frame_area) * 100
                detections.append(((x1, y1, x2, y2), conf, proximity_pct))

    print(f"[OK] Found {len(detections)} snake(s) in image!")

    if not detections:
        print("[WARNING] No snakes detected in the provided image.")
        return

    for idx, ((x1, y1, x2, y2), det_conf, prox_pct) in enumerate(detections, 1):
        bw = x2 - x1
        bh = y2 - y1

        # 2. Crop Snake for Classifier
        pad_x = int(bw * CLASSIFIER_CROP_PADDING)
        pad_y = int(bh * CLASSIFIER_CROP_PADDING)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(frame_w, x2 + pad_x)
        cy2 = min(frame_h, y2 + pad_y)

        snake_crop = frame[cy1:cy2, cx1:cx2]
        if snake_crop.size == 0:
            continue

        print(f"\n--- Snake #{idx} Analysis ---")
        print(f"  Stage 1 Detection Conf: {det_conf*100:.1f}%")
        print(f"  Frame Area / Proximity: {prox_pct:.2f}%")

        # 3. Run Species Classifier
        cls_results = classifier(snake_crop, verbose=False)
        probs = cls_results[0].probs
        top1_id = int(probs.top1)
        top1_conf = float(probs.top1conf)
        species_raw = classifier.names[top1_id]

        clean_name, common_name, hazard_level, is_venomous = analyze_species(species_raw)

        print(f"  Stage 2 Top-1 Species: {common_name} ({clean_name})")
        print(f"  Confidence: {top1_conf*100:.1f}%")
        print(f"  Medical Hazard Level: {hazard_level} ({'🚨 DEADLY VENOMOUS' if is_venomous else '🌿 HARMLESS / MILD'})")

        print("\n  Top 5 Predicted Species Candidates:")
        top5_ids = probs.top5
        for rank, tid in enumerate(top5_ids, 1):
            s_name = classifier.names[int(tid)]
            c_conf = float(probs.data[tid])
            c_clean, c_cname, c_haz, _ = analyze_species(s_name)
            badge = "🚨 DEADLY" if c_haz == 'DEADLY' else ("⚠️ MILD" if c_haz == 'MILD' else "🌿 HARMLESS")
            print(f"    {rank}. {c_cname} [{badge}] - {c_conf*100:.2f}%")

        # Draw on image
        is_close = prox_pct >= PROXIMITY_ALERT_THRESHOLD
        box_color = (0, 0, 255) if hazard_level == 'DEADLY' else (0, 140, 255) if is_close else (0, 220, 220)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

        label_title = f"[{hazard_level}] {common_name}"
        label_stats = f"Det: {int(det_conf*100)}% | Cls: {int(top1_conf*100)}% | Area: {prox_pct:.1f}%"
        
        cv2.putText(frame, label_title, (x1, max(y1 - 25, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        cv2.putText(frame, label_stats, (x1, max(y1 - 6, 45)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if save_output:
        out_name = f"result_{os.path.basename(image_path)}"
        cv2.imwrite(out_name, frame)
        print(f"\n[SAVED] Annotated result written to '{out_name}'")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_pipeline_on_image(sys.argv[1])
    else:
        print("Usage: python test_on_image.py <path_to_snake_image.jpg>")
