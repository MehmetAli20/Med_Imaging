from flask import Flask, request, render_template, url_for, jsonify
import cv2
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tempfile
import shutil

app = Flask(__name__, static_folder='static')
os.makedirs('static', exist_ok=True)
os.makedirs('models', exist_ok=True)

class MedicalImageAnalyzer:
    def __init__(self):
        self.model = None
        self.model_path = 'models/medical_image_model.h5'
        self.load_model()
        
    def load_model(self):
        """Modeli yükle"""
        try:
            self.model = load_model(self.model_path)
            print("Model başarıyla yüklendi.")
        except:
            print("Model yüklenemedi. Lütfen önce modeli eğitin.")
            self.model = None
    
    def preprocess_image(self, image):
        """Görüntüyü ön işleme"""
        # Gri tonlamalı görüntüyü RGB'ye çevir
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Boyutlandırma
        image = cv2.resize(image, (224, 224))
        
        # Normalizasyon
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def analyze_image(self, image_path):
        """Görüntüyü analiz et"""
        if self.model is None:
            return {
                'error': 'Model yüklenemedi. Lütfen önce modeli eğitin.'
            }
        
        # Görüntüyü oku
        image = cv2.imread(image_path)
        if image is None:
            return {
                'error': 'Görüntü okunamadı.'
            }
        
        # Görüntüyü ön işle
        processed_image = self.preprocess_image(image)
        
        # Tahmin yap
        prediction = self.model.predict(np.expand_dims(processed_image, axis=0))[0][0]
        
        # Yoğunluk analizi
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Yoğunluk aralıkları
        density_ranges = {
            'çok düşük': np.sum(hist[:50]),
            'düşük': np.sum(hist[50:100]),
            'orta': np.sum(hist[100:200]),
            'yüksek': np.sum(hist[200:])
        }
        
        # Güven skoru hesapla
        confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
        
        # Sonuçları hazırla
        result = {
            'is_normal': bool(prediction < 0.5),
            'confidence': confidence * 100,
            'density_analysis': density_ranges,
            'prediction_score': float(prediction)
        }
        
        return result
    
    def generate_report(self, analysis_result):
        """Analiz raporu oluştur"""
        if 'error' in analysis_result:
            return analysis_result['error']
        
        report = []
        report.append("TIBBİ GÖRÜNTÜ ANALİZ RAPORU")
        report.append("=" * 30)
        report.append(f"Rapor Tarihi: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        report.append("\nGÖRÜNTÜ KALİTESİ DEĞERLENDİRMESİ:")
        
        # Yoğunluk analizi
        density = analysis_result['density_analysis']
        report.append("\nYoğunluk Dağılımı:")
        report.append(f"- Çok Düşük Yoğunluk: {density['çok düşük']:.2%}")
        report.append(f"- Düşük Yoğunluk: {density['düşük']:.2%}")
        report.append(f"- Orta Yoğunluk: {density['orta']:.2%}")
        report.append(f"- Yüksek Yoğunluk: {density['yüksek']:.2%}")
        
        # Tanı
        report.append("\nTANI:")
        if analysis_result['is_normal']:
            report.append("Normal görüntü")
        else:
            report.append("Anormal görüntü")
        
        report.append(f"Güven Skoru: {analysis_result['confidence']:.1f}%")
        
        # Öneriler
        report.append("\nÖNERİLER:")
        if analysis_result['is_normal']:
            report.append("1. Rutin takip önerilir.")
            report.append("2. Herhangi bir şikayet olması durumunda tekrar değerlendirme yapılabilir.")
        else:
            report.append("1. Detaylı klinik değerlendirme önerilir.")
            report.append("2. Gerekirse ek görüntüleme yöntemleri kullanılabilir.")
            report.append("3. Uzman hekim konsültasyonu önerilir.")
        
        return "\n".join(report)

def plot_image_analysis(image_path, analysis_result):
    """Görüntü analizini görselleştir"""
    # Görüntüyü oku
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Görüntüyü iyileştir
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Histogram hesapla
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_enhanced = cv2.calcHist([enhanced], [0], None, [256], [0, 256])
    
    # Görselleştirme
    plt.figure(figsize=(15, 10))
    
    # Orijinal görüntü
    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Orijinal Görüntü')
    plt.axis('off')
    
    # İyileştirilmiş görüntü
    plt.subplot(2, 2, 2)
    plt.imshow(enhanced, cmap='gray')
    plt.title('İyileştirilmiş Görüntü')
    plt.axis('off')
    
    # Orijinal histogram
    plt.subplot(2, 2, 3)
    plt.plot(hist)
    plt.title('Orijinal Histogram')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    
    # İyileştirilmiş histogram
    plt.subplot(2, 2, 4)
    plt.plot(hist_enhanced)
    plt.title('İyileştirilmiş Histogram')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    
    # Tanı bilgisi
    diagnosis = "Normal" if analysis_result['is_normal'] else "Anormal"
    plt.suptitle(f'Görüntü Analizi - Tanı: {diagnosis} (Güven: {analysis_result["confidence"]:.1f}%)',
                fontsize=16)
    
    # Grafiği kaydet
    plt.tight_layout()
    plt.savefig('static/analysis_plot.png')
    plt.close()

# Flask uygulaması
analyzer = MedicalImageAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'})
    
    # Geçici dosya oluştur
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'temp_image.jpg')
    
    try:
        # Dosyayı kaydet
        file.save(temp_path)
        
        # Görüntüyü analiz et
        analysis_result = analyzer.analyze_image(temp_path)
        
        if 'error' in analysis_result:
            return jsonify(analysis_result)
        
        # Rapor oluştur
        report = analyzer.generate_report(analysis_result)
        
        # Görselleştirme
        plot_image_analysis(temp_path, analysis_result)
        
        return jsonify({
            'report': report,
            'is_normal': analysis_result['is_normal'],
            'confidence': analysis_result['confidence'],
            'plot_url': '/static/analysis_plot.png'
        })
        
    except Exception as e:
        return jsonify({'error': f'Analiz sırasında bir hata oluştu: {str(e)}'})
    finally:
        # Geçici dosyaları temizle
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)
