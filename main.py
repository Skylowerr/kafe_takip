import time
import cv2
from config import INPUT, OUTPUT, LINE_P1, LINE_P2, RED, WHITE, PINK, GREEN, DEVICE
from models import load_models
from classifier import classify_person

# ==================================================
# MODELLERİ YÜKLE
# ==================================================
model, gender_model, appearance_model = load_models()

# ==================================================
# VİDEOYU AÇ
# ==================================================

cap = cv2.VideoCapture(INPUT)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Video açılamadı: {INPUT}"
    )

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    video_fps,
    (width, height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError("Çıktı videosu oluşturulamadı.")

# Takip ve sayım kayıtları
labels = {}
previous_positions = {}
counted = set()

entered = 0
exited = 0
frame_no = 0
start_time = time.time()

line_y = (LINE_P1[1] + LINE_P2[1]) / 2
line_xmin = min(LINE_P1[0], LINE_P2[0])
line_xmax = max(LINE_P1[0], LINE_P2[0])

print("Video işleniyor: TÜM VİDEO")
print(f"Çözünürlük: {width}x{height}")
print(f"Video FPS: {video_fps:.2f}")

# ==================================================
# VİDEOYU KARE KARE İŞLE
# ==================================================

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_no += 1

        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.12,
            imgsz=1280,
            device=DEVICE,
            verbose=False
        )[0]

        women_now = 0
        men_now = 0

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # ----------------------------------
                # KAPI GEÇİŞİ SAYIMI
                # ----------------------------------

                if tid in previous_positions:
                    px, py = previous_positions[tid]

                    if (py - line_y) * (cy - line_y) < 0:
                        ratio = (line_y - py) / (cy - py)
                        cross_x = px + ratio * (cx - px)

                        inside_door = (
                            line_xmin <= cross_x <= line_xmax
                        )

                        if inside_door:
                            if (
                                py < line_y < cy
                                and (tid, "in") not in counted
                            ):
                                entered += 1
                                counted.add((tid, "in"))

                            elif (
                                py > line_y > cy
                                and (tid, "out") not in counted
                            ):
                                exited += 1
                                counted.add((tid, "out"))

                previous_positions[tid] = (cx, cy)

                # ----------------------------------
                # DIŞ MEKÂNI FİLTRELE
                # ----------------------------------
                if cy <= line_y:
                    continue

                # ----------------------------------
                # KİŞİYİ SINIFLANDIR
                # ----------------------------------

                if tid not in labels:
                    labels[tid] = classify_person(
                        frame,
                        (x1, y1, x2, y2),
                        appearance_model,
                        gender_model
                    )

                gender, color = labels[tid]

                if gender == "KADIN":
                    women_now += 1
                else:
                    men_now += 1

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"ID:{tid} | {gender}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        # ==========================================
        # GÖRSEL PANEL VE FPS
        # ==========================================

        cv2.line(
            frame,
            LINE_P1,
            LINE_P2,
            RED,
            3
        )

        elapsed = time.time() - start_time
        processing_fps = frame_no / max(elapsed, 1e-6)

        cv2.putText(
            frame,
            f"KADIN: {women_now}",
            (25, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            PINK,
            2
        )

        cv2.putText(
            frame,
            f"ERKEK: {men_now}",
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            GREEN,
            2
        )

        cv2.putText(
            frame,
            f"Giren: {entered} | Cikan: {exited}",
            (25, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            RED,
            2
        )

        cv2.putText(
            frame,
            f"Isleme FPS: {processing_fps:.1f}",
            (25, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WHITE,
            2
        )

        out.write(frame)

        if frame_no % max(1, int(video_fps * 5)) == 0:
            if total_frames > 0:
                progress = f"{frame_no}/{total_frames}"
            else:
                progress = str(frame_no)

            print(
                f"Kare: {progress} | "
                f"Video süresi: {frame_no / video_fps:.1f} sn | "
                f"İşleme FPS: {processing_fps:.1f}"
            )

finally:
    cap.release()
    out.release()

# ==================================================
# SONUÇ
# ==================================================

print("\nİşlem tamamlandı!")
print(f"İşlenen kare: {frame_no}")
print(f"İşlenen video süresi: {frame_no / video_fps:.1f} saniye")
print(f"Giren kişi: {entered}")
print(f"Çıkan kişi: {exited}")
print(f"Çıktı dosyası: {OUTPUT}")
