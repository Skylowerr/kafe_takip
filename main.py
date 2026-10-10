import os
import time
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline

# ==================================================
# 1. AYARLAR
# ==================================================

INPUT = "input_videos/kisi_sayimi.mp4"
OUTPUT = "outputs/output_results18.mp4"

# Kapı geçiş çizgisi
LINE_P1 = (820, 107)
LINE_P2 = (1075, 108)

# OpenCV renkleri (BGR)
PINK = (180, 105, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

DEVICE = 0 if torch.cuda.is_available() else "cpu"
HF_DEVICE = 0 if torch.cuda.is_available() else -1

os.makedirs("outputs", exist_ok=True)

# ==================================================
# 2. MODELLERİ YÜKLE
# ==================================================

print("YOLOv8x yükleniyor...")
model = YOLO("yolov8x.pt")

print("Kadın/erkek sınıflandırıcısı yükleniyor...")
gender_model = pipeline(
    "image-classification",
    model="prithivMLmods/Realistic-Gender-Classification",
    device=HF_DEVICE
)

print("Başörtüsü ve saç görünümü modeli yükleniyor...")
appearance_model = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
    device=HF_DEVICE
)

# ==================================================
# 3. KADIN / ERKEK SINIFLANDIRMASI
# ==================================================

def classify_person(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]

    # Görüntü sınırlarının dışına çıkma
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    box_height = y2 - y1

    # Baş bölgesi ve üst gövde
    head_end = min(
        y2, y1 + max(1, int(box_height * 0.45))
    )
    upper_end = min(
        y2, y1 + max(1, int(box_height * 0.70))
    )

    head = frame[y1:head_end, x1:x2]
    upper = frame[y1:upper_end, x1:x2]

    if head.size == 0 or upper.size == 0:
        return "ERKEK", GREEN

    # Küçük ve uzaktaki kişi kırpımlarını büyüt
    def to_pil(crop):
        ch, cw = crop.shape[:2]

        scale = min(
            4.0,
            max(1.0, 96.0 / max(1, min(ch, cw)))
        )

        crop = cv2.resize(
            crop,
            (max(1, int(cw * scale)),
             max(1, int(ch * scale))),
            interpolation=cv2.INTER_CUBIC
        )

        return Image.fromarray(
            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        )

    try:
        # Başörtüsü ve arkadan bağlı uzun saç dahil görünüm ipuçları
        appearance_labels = [
            "a woman wearing a hijab or headscarf",
            "a woman with uncovered hair",
            "a bald man",
            "a man with uncovered hair",
            "a woman with long hair tied back in a ponytail, viewed from behind",
            "a woman with long hair tied in a bun, viewed from behind"
        ]

        hp = appearance_model(
            to_pil(head),
            candidate_labels=appearance_labels
        )

        # Başörtüsü veya arkadan bağlı uzun saç
        # güçlü eşleşirse KADIN kategorisini kullan.
        female_appearance_labels = {
            appearance_labels[0],
            appearance_labels[4],
            appearance_labels[5]
        }

        best_label = hp[0]["label"]
        best_score = float(hp[0]["score"])
        second_score = (
            float(hp[1]["score"]) if len(hp) > 1 else 0.0
        )

        if (
            best_label in female_appearance_labels
            and best_score >= 0.30
            and best_score - second_score >= 0.03
        ):
            return "KADIN", PINK

        # Görsel ipucu yeterince belirgin değilse
        # genel kadın/erkek sınıflandırıcısını kullan.
        gp = gender_model(to_pil(upper), top_k=None)

        female_score = 0.0
        male_score = 0.0

        for item in gp:
            label = item["label"].lower()
            score = float(item["score"])

            if any(word in label for word in (
                "female", "woman", "girl",
                "kadin", "kadın", "kız", "kiz"
            )):
                female_score = max(female_score, score)

            elif any(word in label for word in (
                "male", "man", "boy", "erkek"
            )):
                male_score = max(male_score, score)

        if female_score > male_score:
            return "KADIN", PINK

        return "ERKEK", GREEN

    except Exception as e:
        print("Sınıflandırma hatası:", e)
        return "ERKEK", GREEN


# ==================================================
# 4. VİDEOYU AÇ
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

# Videonun tamamı işlenecek; süre sınırı yok.
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
# 5. VİDEOYU KARE KARE İŞLE
# ==================================================

try:
    while True:
        ok, frame = cap.read()

        # Video sonuna gelince dur
        if not ok:
            break

        frame_no += 1

        # YOLOv8x + yerleşik ByteTrack
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

                # Kişinin merkez noktası
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # ----------------------------------
                # KAPI GEÇİŞİ SAYIMI
                # ----------------------------------

                if tid in previous_positions:
                    px, py = previous_positions[tid]

                    # Önceki ve mevcut konum çizginin
                    # farklı taraflarında mı?
                    if (py - line_y) * (cy - line_y) < 0:
                        ratio = (line_y - py) / (cy - py)
                        cross_x = px + ratio * (cx - px)

                        inside_door = (
                            line_xmin <= cross_x <= line_xmax
                        )

                        if inside_door:
                            # Yukarıdan aşağıya: giriş
                            if (
                                py < line_y < cy
                                and (tid, "in") not in counted
                            ):
                                entered += 1
                                counted.add((tid, "in"))

                            # Aşağıdan yukarıya: çıkış
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
                # Bu ayar çizginin altını kafe içi sayar.
                if cy <= line_y:
                    continue

                # ----------------------------------
                # KİŞİYİ SINIFLANDIR
                # Her takip ID'sini bir kez sınıflandır.
                # ----------------------------------

                if tid not in labels:
                    labels[tid] = classify_person(
                        frame,
                        (x1, y1, x2, y2)
                    )

                gender, color = labels[tid]

                if gender == "KADIN":
                    women_now += 1
                else:
                    men_now += 1

                # Kişi kutusu ve etiket
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
        # 6. GÖRSEL PANEL VE FPS
        # ==========================================

        # Kapı çizgisi
        cv2.line(
            frame,
            LINE_P1,
            LINE_P2,
            RED,
            3
        )

        # Gerçek işleme hızı; videonun kayıt FPS'i değil.
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

        # İşlenmiş kareyi video dosyasına yaz
        out.write(frame)

        # Konsola her 5 saniyelik video ilerlemesinde bilgi yaz
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
# 7. SONUÇ
# ==================================================

print("\nİşlem tamamlandı!")
print(f"İşlenen kare: {frame_no}")
print(f"İşlenen video süresi: {frame_no / video_fps:.1f} saniye")
print(f"Giren kişi: {entered}")
print(f"Çıkan kişi: {exited}")
print(f"Çıktı dosyası: {OUTPUT}")
