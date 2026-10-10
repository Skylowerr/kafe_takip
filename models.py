from ultralytics import YOLO
from transformers import pipeline
from config import HF_DEVICE

# ==================================================
# MODELLERİ YÜKLE
# ==================================================

def load_models():
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

    return model, gender_model, appearance_model
