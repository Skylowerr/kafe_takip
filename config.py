import os
import torch

# ==================================================
# AYARLAR
# ==================================================

INPUT = "input_videos/kisi_sayimi.mp4"
OUTPUT = "outputs/output_results.mp4"

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
