# Tıbbi Görüntü Analizi

Bu proje, derin öğrenme kullanarak tıbbi görüntüleri (özellikle akciğer röntgenleri) analiz eden bir web uygulamasıdır. ResNet50V2 mimarisi kullanılarak geliştirilmiş bir model ile normal ve anormal görüntüleri sınıflandırır.

## Özellikler

- Derin öğrenme tabanlı görüntü analizi
- Modern ve kullanıcı dostu web arayüzü
- Detaylı analiz raporları
- Görüntü kalitesi değerlendirmesi
- Yoğunluk analizi ve histogram görselleştirme

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn flask
```

2. Modeli eğitin:
```bash
python train_model.py
```

3. Web uygulamasını başlatın:
```bash
python main.py
```

## Kullanım

1. Web tarayıcınızda `http://localhost:5000` adresine gidin
2. Analiz etmek istediğiniz görüntüyü yükleyin
3. "Analiz Et" butonuna tıklayın
4. Sonuçları ve detaylı raporu görüntüleyin

## Proje Yapısı

- `main.py`: Web uygulaması ve görüntü analiz mantığı
- `train_model.py`: Model eğitimi ve veri işleme
- `templates/`: HTML şablonları
- `static/`: Statik dosyalar (CSS, JavaScript, görüntüler)
- `models/`: Eğitilmiş model dosyaları

## Model Detayları

- Mimarisi: ResNet50V2 (transfer learning)
- Giriş boyutu: 224x224x3
- Sınıflandırma: Binary (Normal/Anormal)
- Veri artırma: Döndürme, kaydırma, kesme, yakınlaştırma

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.
