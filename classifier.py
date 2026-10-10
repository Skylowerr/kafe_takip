import cv2
from PIL import Image
from config import PINK, GREEN

# ==================================================
# KADIN / ERKEK SINIFLANDIRMASI
# ==================================================

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

def classify_person(frame, box, appearance_model, gender_model):
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
