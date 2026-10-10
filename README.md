<div align="center">

# 🎯 AI Destekli Kişi Sayımı ve Demografik Analiz Sistemi

Güvenlik kameraları ve video kayıtları üzerinden **kişi sayımı (giriş/çıkış)** ve **cinsiyet/görünüm analizi** yapan modüler bir Python uygulaması.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗-Transformers-FFD21E)

</div>

---

## 📖 Hakkında

Bu proje; **YOLOv8**, **ByteTrack** ve **Hugging Face Transformers** modellerini birleştirerek video üzerindeki kişileri tespit eder, kararlı biçimde takip eder, sanal bir kapı çizgisinden geçişlerini sayar ve her kişi için görünüm özelliklerini (başörtüsü, saç stili, cinsiyet vb.) sınıflandırır.

## 🚀 Öne Çıkan Özellikler

- **Yüksek Doğruluklu Takip:** YOLOv8x ve ByteTrack entegrasyonu ile kişilerin `ID` bazlı, kararlı takibi.
- **Detaylı Görünüm Analizi:** CLIP (zero-shot) ve özel sınıflandırma modelleriyle başörtüsü, arkadan bağlı uzun saç ve genel cinsiyet tespiti.
- **Giriş/Çıkış Sayımı:** Belirlenen sanal kapı çizgisini geçen kişilerin yön analiziyle otomatik sayımı.
- **Modüler Mimari:** İşlevlerine göre ayrılmış dosya yapısı sayesinde kolay bakım ve geliştirme.

## 📂 Proje Yapısı

```text
.
├── config.py         # Dosya yolları, kapı koordinatları ve renk paletleri
├── models.py         # YOLOv8 ve Hugging Face modellerinin yüklenme mantığı
├── classifier.py     # Kişi kırpma, ön işleme ve sınıflandırma fonksiyonları
└── main.py           # Ana video işleme döngüsü ve sayım algoritması
```

## 🛠️ Kurulum

### 1. Depoyu klonlayın

```bash
git clone https://github.com/<Skylowerr>/<kafe_takip>.git
cd <kafe_takip>
```

### 2. (Önerilen) Sanal ortam oluşturun

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Bağımlılıkları yükleyin

```bash
pip install opencv-python torch torchvision torchaudio ultralytics transformers pillow
```

> 💡 **Not:** CUDA destekli bir ekran kartınız (GPU) varsa PyTorch otomatik olarak GPU hızlandırmasını kullanır.

## ⚙️ Yapılandırma

Tüm ayarlar `config.py` dosyasından yapılır:

| Parametre | Açıklama | Örnek |
|-----------|----------|-------|
| `INPUT` | Analiz edilecek kaynak video yolu | `input_videos/kisi_sayimi.mp4` |
| `OUTPUT` | İşlenmiş videonun kaydedileceği yol | `outputs/output_results18.mp4` |
| `LINE_P1` / `LINE_P2` | Kapı çizgisinin başlangıç ve bitiş koordinatları | `(x1, y1)` / `(x2, y2)` |

> Not: Giriş videosunu `input_videos/` klasörüne, çıktılar için de `outputs/` klasörünün var olduğundan emin olun.

## ▶️ Kullanım

```bash
python main.py
```

Çalışma sırasında konsola düzenli aralıklarla **ilerleme durumu** ve **işleme hızı (FPS)** yazdırılır. İşlem bittiğinde sonuç videosu `outputs/` klasöründe hazır olur.

## 🔄 Nasıl Çalışır?

1. **Tespit:** YOLOv8x her karedeki kişileri bulur.
2. **Takip:** ByteTrack her kişiye kalıcı bir `ID` atar.
3. **Sayım:** Kişinin merkez noktasının kapı çizgisini hangi yönde geçtiği izlenir; giriş/çıkış sayaçları güncellenir.
4. **Sınıflandırma:** Kişi kırpılıp ön işlemden geçirilir; CLIP ve özel modellerle görünüm özellikleri tahmin edilir.
5. **Çıktı:** Kutular, etiketler, kapı çizgisi ve sayaçlar kareye çizilerek video olarak kaydedilir.

## ⚠️ Önemli Notlar

- Cinsiyet ve görünüm tahminleri **olasılıksal** sonuçlardır; kamera açısı, ışık ve çözünürlüğe göre hatalı olabilir.
- Kişisel veri içeren görüntülerin işlenmesi **KVKK / GDPR** gibi mevzuatlara tabidir. Projeyi gerçek ortamlarda kullanmadan önce gerekli izin ve aydınlatma süreçlerini yerine getirin.
- Bu proje eğitim ve araştırma amaçlıdır; kişileri profillemek veya ayrımcılık yapmak için kullanılmamalıdır.

## 🙏 Teşekkürler

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [OpenAI CLIP](https://github.com/openai/CLIP)
