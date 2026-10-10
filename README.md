# KafeTakip
AI Destekli Kişi Sayımı ve Demografik Analiz Sistemi
Bu proje; güvenlik kameraları veya video kayıtları üzerinden YOLOv8, ByteTrack ve Hugging Face Transformers modellerini kullanarak kişi sayımı (giriş/çıkış) ve gelişmiş cinsiyet/görünüm analizi (başörtüsü, saç stili vb.) gerçekleştiren modüler bir Python uygulamasıdır.

🚀 Öne Çıkan Özellikler
Yüksek Doğruluklu Takip: YOLOv8x modeli ve ByteTrack entegrasyonu ile kişilerin kararlı bir şekilde takip edilmesi (ID bazlı).

Detaylı Görünüm Analizi: CLIP (Zero-shot) ve özel sınıflandırma modelleri kullanılarak başörtüsü, arkadan bağlı uzun saç ve genel cinsiyet tespiti.

Geçiş (Giriş/Çıkış) Sayımı: Belirlenen sanal kapı çizgisinden geçen kişilerin yön analizi ile otomatik olarak sayılması.

Modüler Mimari: Kolay bakım ve geliştirme için işlevlerine göre ayrılmış dosya yapısı.

📂 Proje Dosya Yapısı
Plaintext
.
├── config.py         # Dosya yolları, kapı koordinatları ve renk paletleri
├── models.py         # YOLOv8 ve Hugging Face modellerinin yüklenme mantığı
├── classifier.py     # Kişi kırpma, ön işleme ve sınıflandırma fonksiyonları
└── main.py           # Ana video işleme döngüsü ve sayım algoritması
🛠️ Gereksinimler ve Kurulum
Projeyi çalıştırmadan önce gerekli Python kütüphanelerini yüklemeniz gerekir:

Bash
pip install opencv-python torch torchvision torchaudio ultralytics transformers pillow
Not: CUDA destekli bir ekran kartınız (GPU) varsa, PyTorch otomatik olarak GPU hızlandırmasını aktif hale getirecektir.

⚙️ Yapılandırma (config.py)
Giriş videosu yolunu, çıktı dosyasını ve kapı geçiş çizgisinin koordinatlarını config.py dosyasından düzenleyebilirsiniz:

INPUT: Analiz edilecek kaynak video yolu (input_videos/kisi_sayimi.mp4)

OUTPUT: İşlenmiş sonucun kaydedileceği video yolu (outputs/output_results18.mp4)

LINE_P1 / LINE_P2: Kapı çizgisinin başlangıç ve bitiş koordinatları

🚀 Kullanım
Sistemi başlatmak için ana dosyanızı çalıştırmanız yeterlidir:

Bash
python main.py
Çalışma sırasında konsola düzenli aralıklarla ilerleme durumu ve işleme hızı (FPS) yazdırılacaktır. İşlem bittikten sonra sonuç videosu outputs/ klasörü altında hazır olacaktır.
