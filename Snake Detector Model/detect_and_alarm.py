import cv2
import time
import os
import threading
import winsound
import numpy as np
from collections import deque, Counter
from ultralytics import RTDETR, YOLO

# ==============================================================================
# 1. CONFIGURATION & MODEL PATHS
# ==============================================================================
DETECTOR_PATH = "best_snake_hunter_rtdetr.pt"
CLASSIFIER_PATH = "best_snake_teller_498cls.pt"

ENABLE_ARDUINO = False     # Set to True when a physical Arduino is plugged in
ARDUINO_PORT = 'COM3'      # Arduino COM port if enabled
BAUD_RATE = 9600

# Proximity threshold: percentage of webcam frame occupied by the snake bounding box
# e.g., > 3.5% area means the snake is dangerously close to the camera/user
PROXIMITY_ALERT_THRESHOLD = 3.5  

# Confidence thresholds
DETECTION_CONF_THRESHOLD = 0.40   # RT-DETR snake detection threshold
CLASSIFIER_CROP_PADDING = 0.12    # 12% padding around detected snake for classifier

# Audio cooldown in seconds to prevent sound overlap
ALARM_COOLDOWN = 1.4

# ==============================================================================
# 2. MEDICAL & VENOM KNOWLEDGE BASE (498 SPECIES)
# ==============================================================================
# Highly dangerous venomous genera (Elapids, Vipers, Pit Vipers, Atractaspids)
VENOMOUS_GENERA = {
    # Elapidae (Cobras, Kraits, Mambas, Coral Snakes, Sea Snakes, Taipans, etc.)
    'naja', 'ophiophagus', 'bungarus', 'dendroaspis', 'micrurus', 'micruroides',
    'acanthophis', 'oxyuranus', 'pseudonaja', 'notechis', 'hydrophis', 'laticauda',
    'aipysurus', 'emydocephalus', 'hemachatus', 'aspidelaps', 'walterinnesia',
    'calliophis', 'demansia', 'drysdalia', 'furina', 'hoplocephalus', 'tropidechis',
    
    # Viperidae (Vipers, Pit Vipers, Rattlesnakes, Adders, Moccasins)
    'vipera', 'daboia', 'echis', 'bitis', 'cerastes', 'crotalus', 'sistrurus',
    'agkistrodon', 'bothrops', 'bothriechis', 'bothrocophias', 'trimeresurus',
    'gloydius', 'tropidolaemus', 'lachesis', 'calloselasma', 'deinagkistrodon',
    'protobothrops', 'hypnale', 'ovophis', 'atheris', 'pseudocerastes',
    'macrovipera', 'montivipera', 'causus', 'cerrophidion', 'metlapilcoatlus',
    'porthidium',
    
    # Atractaspididae (Burrowing Asps / Stiletto Snakes)
    'atractaspis', 'macrelaps',
    
    # Medically significant Rear-Fanged / Colubridae (can cause fatalities)
    'dispholidus', 'thelotornis', 'rhabdophis', 'tachymenis'
}

# Mildly venomous / Opisthoglyphous (generally non-lethal to humans)
MILDLY_VENOMOUS_GENERA = {
    'boiga', 'malpolon', 'philodryas', 'ahaetulla', 'chrysopelea',
    'heterodon', 'psammodynastes', 'psammophis', 'telescopus', 'trimorphodon'
}

# Popular / Common English names for iconic species
COMMON_NAMES = {
    # Big Four & Famous Asian Venomous
    "Naja_naja": "Indian Cobra / Spectacled Cobra",
    "Naja_kaouthia": "Monocled Cobra",
    "Naja_atra": "Chinese Cobra",
    "Naja_siamensis": "Indochinese Spitting Cobra",
    "Ophiophagus_hannah": "King Cobra",
    "Daboia_russelii": "Russell's Viper",
    "Bungarus_caeruleus": "Common Krait",
    "Bungarus_fasciatus": "Banded Krait",
    "Bungarus_candidus": "Malayan Krait",
    "Bungarus_multicinctus": "Many-banded Krait",
    "Echis_carinatus": "Saw-scaled Viper",
    
    # African Deadly Snakes
    "Dendroaspis_polylepis": "Black Mamba",
    "Dendroaspis_angusticeps": "Eastern Green Mamba",
    "Bitis_gabonica": "Gaboon Viper",
    "Bitis_arietans": "Puff Adder",
    "Bitis_caudalis": "Horned Adder",
    "Naja_nigricollis": "Black-necked Spitting Cobra",
    "Naja_mossambica": "Mozambique Spitting Cobra",
    "Naja_nivea": "Cape Cobra",
    "Dispholidus_typus": "Boomslang",
    "Thelotornis_capensis": "Twig / Vine Snake",
    
    # American Rattlesnakes & Pit Vipers
    "Crotalus_atrox": "Western Diamondback Rattlesnake",
    "Crotalus_adamanteus": "Eastern Diamondback Rattlesnake",
    "Crotalus_horridus": "Timber Rattlesnake",
    "Crotalus_cerastes": "Sidewinder Rattlesnake",
    "Crotalus_viridis": "Prairie Rattlesnake",
    "Crotalus_durissus": "South American Rattlesnake",
    "Crotalus_scutulatus": "Mojave Rattlesnake",
    "Agkistrodon_contortrix": "Eastern Copperhead",
    "Agkistrodon_piscivorus": "Northern Cottonmouth",
    "Bothrops_asper": "Terciopelo / Fer-de-Lance",
    "Bothrops_atrox": "Common Lancehead",
    "Bothriechis_schlegelii": "Eyelash Palm Pit Viper",
    
    # Pythons, Boas & Non-Venomous Giants
    "Python_molurus": "Indian Rock Python",
    "Python_bivittatus": "Burmese Python",
    "Python_sebae": "African Rock Python",
    "Malayopython_reticulatus": "Reticulated Python",
    "Boa_constrictor": "Boa Constrictor",
    "Boa_imperator": "Central American Boa",
    "Eunectes_murinus": "Green Anaconda",
    "Ptyas_mucosa": "Oriental Rat Snake",
    "Pantherophis_guttatus": "Corn Snake",
    "Pantherophis_obsoletus": "Western Rat Snake",
    "Lampropeltis_getula": "Eastern Kingsnake",
    "Natrix_natrix": "European Grass Snake",
    "Natrix_tessellata": "Dice Snake",
    "Coronella_austriaca": "Smooth Snake",
    "Thamnophis_sirtalis": "Common Garter Snake",
    "Heterodon_platirhinos": "Eastern Hognose Snake",
}

def analyze_species(species_raw_name):
    """
    Returns (clean_name, common_name, hazard_level, is_venomous)
    hazard_level: 'DEADLY', 'MILD', 'HARMLESS'
    """
    genus = species_raw_name.split('_')[0].lower()
    clean_name = species_raw_name.replace('_', ' ')
    common_name = COMMON_NAMES.get(species_raw_name, clean_name)
    
    if genus in VENOMOUS_GENERA:
        return clean_name, common_name, 'DEADLY', True
    elif genus in MILDLY_VENOMOUS_GENERA:
        return clean_name, common_name, 'MILD', False
    else:
        return clean_name, common_name, 'HARMLESS', False

# ==============================================================================
# 3. HARDWARE & AUDIO ALARMS
# ==============================================================================
arduino = None
if ENABLE_ARDUINO:
    try:
        import serial
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"[OK] Connected to Arduino on {ARDUINO_PORT}")
    except Exception as e:
        print(f"[WARNING] Could not connect to Arduino: {e}. Falling back to PC speaker.")
        arduino = None

sound_muted = False

def play_pc_buzzer(hazard_level):
    """Plays distinct audible sirens in background without freezing video."""
    if sound_muted:
        return
    try:
        if hazard_level == 'DEADLY':
            # VENOMOUS EMERGENCY: 3 urgent high-frequency siren beeps
            for _ in range(3):
                winsound.Beep(2900, 110)
                time.sleep(0.03)
        elif hazard_level == 'MILD':
            # MILD WARNING: 2 medium-pitch pulses
            winsound.Beep(2000, 180)
            time.sleep(0.04)
            winsound.Beep(2000, 180)
        else:
            # NON-VENOMOUS / HARMLESS: 1 gentle low notification chime
            winsound.Beep(1400, 260)
    except Exception:
        pass

# ==============================================================================
# 4. LOAD RESEARCH MODELS
# ==============================================================================
print("\n" + "="*65)
print("  INITIALIZING TWO-STAGE SNAKE DETECTION & SPECIES TELLER")
print("="*65)

if not os.path.exists(DETECTOR_PATH):
    raise FileNotFoundError(f"[ERROR] Stage 1 detector weights not found: '{DETECTOR_PATH}'")
if not os.path.exists(CLASSIFIER_PATH):
    raise FileNotFoundError(f"[ERROR] Stage 2 classifier weights not found: '{CLASSIFIER_PATH}'")

print(f"1. Loading Stage 1 Snake Hunter (RT-DETR + PIoU): {DETECTOR_PATH}...")
detector = RTDETR(DETECTOR_PATH)
print("   ✅ Stage 1 Detector Ready (99.08% mAP@0.5)!")

print(f"2. Loading Stage 2 Species Classifier (YOLOv8m-cls 498 Classes): {CLASSIFIER_PATH}...")
classifier = YOLO(CLASSIFIER_PATH)
print(f"   ✅ Stage 2 Classifier Ready ({len(classifier.names)} Global Species)!")

# ==============================================================================
# 5. WEBCAM INITIALIZATION & MAIN LOOP
# ==============================================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam (camera index 0).")
    exit()

print("\n" + "="*65)
print("  SNAKE SAFETY RADAR RUNNING")
print("  - Stage 1: RT-DETR finds snakes in complex camouflage")
print("  - Stage 2: YOLOv8m classifies across 498 global species")
print("  - Controls: Press 'q' to Quit | 'm' to Mute/Unmute | 's' for Screenshot")
print("="*65 + "\n")

last_alarm_time = 0
history_buffer = deque(maxlen=7)  # 7-frame temporal voting stability
prev_fps_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_h * frame_w

    # Calculate FPS
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_fps_time, 1e-4))
    prev_fps_time = now

    # ---------------------------------------------------------
    # STAGE 1: Detect Snake Bounding Boxes using RT-DETR
    # ---------------------------------------------------------
    det_results = detector(frame, conf=DETECTION_CONF_THRESHOLD, verbose=False)
    
    best_snake = None
    max_box_area = 0

    for r in det_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Clamp to frame bounds
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w))
            y2 = max(0, min(y2, frame_h))

            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            box_area = box_w * box_h

            if box_area > max_box_area and box_w > 10 and box_h > 10:
                max_box_area = box_area
                proximity_pct = (box_area / frame_area) * 100
                best_snake = {
                    'box': (x1, y1, x2, y2),
                    'det_conf': conf,
                    'proximity_pct': proximity_pct
                }

    # ---------------------------------------------------------
    # STAGE 2: Crop Snake & Predict Fine-Grained Species
    # ---------------------------------------------------------
    current_detection_data = None

    if best_snake:
        x1, y1, x2, y2 = best_snake['box']
        bw = x2 - x1
        bh = y2 - y1

        # Add padding around the snake crop for context
        pad_x = int(bw * CLASSIFIER_CROP_PADDING)
        pad_y = int(bh * CLASSIFIER_CROP_PADDING)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(frame_w, x2 + pad_x)
        cy2 = min(frame_h, y2 + pad_y)

        snake_crop = frame[cy1:cy2, cx1:cx2]

        if snake_crop.size > 0:
            cls_results = classifier(snake_crop, verbose=False)
            probs = cls_results[0].probs
            top1_id = int(probs.top1)
            top1_conf = float(probs.top1conf)
            species_raw = classifier.names[top1_id]

            # Top 3 candidates for the HUD sidebar
            top3_ids = probs.top5[:3] if hasattr(probs, 'top5') else [top1_id]
            top3_candidates = []
            for tid in top3_ids:
                s_name = classifier.names[int(tid)]
                c_conf = float(probs.data[tid])
                _, c_cname, c_haz, _ = analyze_species(s_name)
                top3_candidates.append((c_cname, c_conf, c_haz))

            clean_name, common_name, hazard_level, is_venomous = analyze_species(species_raw)

            current_detection_data = {
                'box': best_snake['box'],
                'det_conf': best_snake['det_conf'],
                'proximity_pct': best_snake['proximity_pct'],
                'species_raw': species_raw,
                'clean_name': clean_name,
                'common_name': common_name,
                'hazard_level': hazard_level,
                'is_venomous': is_venomous,
                'cls_conf': top1_conf,
                'top3': top3_candidates
            }

    # ---------------------------------------------------------
    # Temporal Voting & Stability
    # ---------------------------------------------------------
    if current_detection_data is not None:
        history_buffer.append(current_detection_data)
    else:
        if len(history_buffer) > 0 and (time.time() - last_alarm_time > 1.2):
            history_buffer.popleft()

    # Determine stable state if buffer has recent votes
    active_display = None
    if len(history_buffer) >= 2:
        # Majority voting on hazard level & species
        hazard_votes = Counter([item['hazard_level'] for item in history_buffer])
        stable_hazard = hazard_votes.most_common(1)[0][0]

        species_votes = Counter([item['common_name'] for item in history_buffer])
        stable_species = species_votes.most_common(1)[0][0]

        # Use the latest detection coordinates and highest confidence
        latest = history_buffer[-1]
        active_display = {
            'box': latest['box'],
            'proximity_pct': latest['proximity_pct'],
            'det_conf': latest['det_conf'],
            'cls_conf': latest['cls_conf'],
            'common_name': stable_species,
            'clean_name': latest['clean_name'],
            'hazard_level': stable_hazard,
            'top3': latest['top3']
        }

    # ---------------------------------------------------------
    # Proximity & Alarm Decision
    # ---------------------------------------------------------
    danger_detected = False
    highest_prox = 0.0

    if active_display:
        highest_prox = active_display['proximity_pct']
        hazard = active_display['hazard_level']
        
        if highest_prox >= PROXIMITY_ALERT_THRESHOLD:
            danger_detected = True

        # Trigger audio and Arduino alarm
        if danger_detected and (now - last_alarm_time > ALARM_COOLDOWN):
            if hazard == 'DEADLY':
                print(f"🚨 [EMERGENCY] DEADLY VENOMOUS SNAKE ({active_display['common_name']}) in close range ({highest_prox:.1f}%)!")
            elif hazard == 'MILD':
                print(f"⚠️ [WARNING] MILDLY VENOMOUS SNAKE ({active_display['common_name']}) close ({highest_prox:.1f}%).")
            else:
                print(f"🌿 [NOTICE] Non-Venomous snake ({active_display['common_name']}) close ({highest_prox:.1f}%).")

            threading.Thread(target=play_pc_buzzer, args=(hazard,), daemon=True).start()

            if arduino:
                sig = b'V' if hazard == 'DEADLY' else (b'M' if hazard == 'MILD' else b'N')
                arduino.write(sig)

            last_alarm_time = now

    # =========================================================
    # VISUAL RENDERING & HEADS-UP DISPLAY (HUD)
    # =========================================================

    # 1. Draw Target Box & Species Card
    if active_display:
        x1, y1, x2, y2 = active_display['box']
        hazard = active_display['hazard_level']

        if danger_detected:
            box_color = (0, 0, 255) if hazard == 'DEADLY' else (0, 140, 255) if hazard == 'MILD' else (0, 220, 220)
        else:
            box_color = (0, 255, 255) if hazard == 'DEADLY' else (0, 255, 100)

        # Draw main bounding box with corner accent brackets
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        corner_len = min(20, (x2 - x1)//3, (y2 - y1)//3)
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, 4)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, 4)
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, 4)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, 4)
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, 4)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, 4)
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 4)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, 4)

        # Species label card above the snake
        hazard_badge = "VENOMOUS" if hazard == 'DEADLY' else "MILD VENOM" if hazard == 'MILD' else "NON-VENOMOUS"
        title_text = f"[{hazard_badge}] {active_display['common_name']}"
        stats_text = f"Det: {int(active_display['det_conf']*100)}% | Cls: {int(active_display['cls_conf']*100)}% | Prox: {highest_prox:.1f}%"

        card_y = max(y1 - 42, 50)
        cv2.rectangle(frame, (x1 - 2, card_y - 20), (x1 + max(len(title_text)*9, 240), card_y + 22), (20, 20, 20), -1)
        cv2.rectangle(frame, (x1 - 2, card_y - 20), (x1 + max(len(title_text)*9, 240), card_y + 22), box_color, 1)

        cv2.putText(frame, title_text, (x1 + 4, card_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_color, 2)
        cv2.putText(frame, stats_text, (x1 + 4, card_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        # Top 3 Candidates Panel (Right-side HUD)
        panel_x = frame_w - 270
        panel_y = 55
        cv2.rectangle(frame, (panel_x, panel_y), (frame_w - 10, panel_y + 115), (25, 25, 25), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (frame_w - 10, panel_y + 115), (70, 70, 70), 1)
        cv2.putText(frame, "TOP SPECIES CANDIDATES", (panel_x + 10, panel_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        for i, (c_name, c_conf, c_haz) in enumerate(active_display['top3']):
            c_color = (0, 0, 255) if c_haz == 'DEADLY' else (0, 180, 255) if c_haz == 'MILD' else (0, 255, 120)
            short_name = (c_name[:22] + '..') if len(c_name) > 24 else c_name
            row_y = panel_y + 42 + (i * 24)
            cv2.putText(frame, f"{i+1}. {short_name}", (panel_x + 10, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
            cv2.putText(frame, f"{int(c_conf*100)}%", (frame_w - 45, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c_color, 1)

    # 2. Top Emergency Banner
    if danger_detected:
        hazard = active_display['hazard_level'] if active_display else 'DEADLY'
        banner_bg = (0, 0, 200) if hazard == 'DEADLY' else (0, 110, 220)
        banner_msg = "!!! ALARM: VENOMOUS SNAKE DANGER CLOSE !!!" if hazard == 'DEADLY' else "!!! WARNING: SNAKE IN CLOSE PROXIMITY !!!"
        cv2.rectangle(frame, (0, 0), (frame_w, 42), banner_bg, -1)
        cv2.putText(frame, banner_msg, (20, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (0, 0), (frame_w, 36), (35, 35, 35), -1)
        cv2.putText(frame, "SNAKE RADAR ACTIVE: SCANNING FOR COMPLEX NATURAL CAMOUFLAGE", 
                    (20, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 150), 1)

    # 3. Bottom Proximity Gauge Bar
    bar_width = 280
    fill_ratio = min(highest_prox / (PROXIMITY_ALERT_THRESHOLD * 2.0), 1.0)
    bar_fill = int(fill_ratio * bar_width)
    bar_color = (0, 0, 255) if (danger_detected and active_display and active_display['hazard_level'] == 'DEADLY') else \
                (0, 140, 255) if danger_detected else (0, 255, 255) if highest_prox > 0 else (100, 100, 100)

    cv2.rectangle(frame, (20, frame_h - 28), (20 + bar_width, frame_h - 14), (50, 50, 50), -1)
    if bar_fill > 0:
        cv2.rectangle(frame, (20, frame_h - 28), (20 + bar_fill, frame_h - 14), bar_color, -1)
    cv2.rectangle(frame, (20, frame_h - 28), (20 + bar_width, frame_h - 14), (120, 120, 120), 1)

    cv2.putText(frame, f"PROXIMITY: {highest_prox:.1f}% (Danger Threshold: {PROXIMITY_ALERT_THRESHOLD}%)", 
                (20, frame_h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # 4. Status HUD Overlay (FPS, Audio state)
    mute_status = "[MUTED]" if sound_muted else "[AUDIO ON]"
    cv2.putText(frame, f"FPS: {fps:.1f} | {mute_status}", (frame_w - 180, frame_h - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # Display Output
    cv2.imshow("Snake Safety Radar - RT-DETR + 498-Species Classifier", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        sound_muted = not sound_muted
        print(f"[AUDIO] Audio alarm {'MUTED' if sound_muted else 'UNMUTED'}")
    elif key == ord('s'):
        snap_path = f"snapshot_snake_{int(time.time())}.jpg"
        cv2.imwrite(snap_path, frame)
        print(f"[SAVED] Saved radar snapshot to '{snap_path}'")

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
