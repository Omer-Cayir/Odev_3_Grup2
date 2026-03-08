import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def calculate_energy(frame):
    """Bir çerçevenin karesel enerjisini hesaplar."""
    # Sinyalin değerlerini float64'e çeviriyoruz ki taşma (overflow) olmasın
    frame = np.array(frame, dtype=np.float64)
    return np.sum(frame**2)

def calculate_zcr(frame):
    """Bir çerçevenin Sıfır Geçiş Oranını (ZCR) hesaplar."""
    # İşaret değişimlerini sayar: Sinyalin bir önceki değeri ile şimdiki değerinin çarpımı negatifse, ekseni kesmiş demektir.
    crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return crossings / len(frame)

def process_audio(file_path):
    # 1. Sinyali Okuma
    try:
        sample_rate, signal = wav.read(file_path)
    except FileNotFoundError:
        print("HATA: 'audio.wav' dosyası bulunamadı. Lütfen kodun bulunduğu klasöre bir ses dosyası ekleyin.")
        return

    # Eğer stereo ise mono'ya çevir (basitlik için sadece bir kanalı al)
    if len(signal.shape) > 1:
        signal = signal[:, 0]
        
    # Parametreler
    frame_length_ms = 20
    overlap_ms = 10
    
    frame_length = int((frame_length_ms / 1000) * sample_rate)
    overlap = int((overlap_ms / 1000) * sample_rate)
    step_size = frame_length - overlap
    
    num_frames = int(np.floor((len(signal) - frame_length) / step_size)) + 1
    
    energies = np.zeros(num_frames)
    zcrs = np.zeros(num_frames)
    
    # 2. Çerçeveleme (Framing), Enerji ve ZCR Hesaplama
    for i in range(num_frames):
        start = i * step_size
        end = start + frame_length
        frame = signal[start:end]
        
        energies[i] = calculate_energy(frame)
        zcrs[i] = calculate_zcr(frame)
        
    # 3. Gürültü Tahmini (İlk 150 ms - konuşma olmadığı varsayımıyla)
    noise_duration_ms = 150
    noise_frames_count = int(noise_duration_ms / overlap_ms) # Zamanı zamana bölüyoruz!
    
    if noise_frames_count > num_frames:
        noise_frames_count = num_frames // 10 # Güvenlik önlemi
        
    noise_energy_mean = np.mean(energies[:noise_frames_count])
    energy_threshold = noise_energy_mean * 0.04 # Eşik değeri: Gürültünün 2.0 katı

    # BURALARI EKLİYORSUN:
    print("\n--- DIAGNOSTIK VERILER ---")
    print(f"Sinyalin Toplam Uzunlugu: {len(signal)} Ornek")
    print(f"Sinyaldeki Maksimum Enerji: {np.max(energies):.4f}")
    print(f"Hesaplanan Gurultu Ortalamasi: {noise_energy_mean:.4f}")
    print(f"Kullanilan Esik Degeri: {energy_threshold:.4f}")
    print("--------------------------\n")
    
    # 4. Karar Maskesi ve Hangover (Akıcılık Kontrolü)
    vad_mask = np.zeros(num_frames, dtype=bool)
    
    # Ön etiketleme (Sadece Enerjiye göre)
    for i in range(num_frames):
        if energies[i] > energy_threshold:
            vad_mask[i] = True
            
    # Hangover Time Uygulaması (3 frame bekleme süresi)
    hangover_frames = 3
    final_vad_mask = np.copy(vad_mask)
    
    for i in range(1, num_frames):
        if not vad_mask[i]: # Eğer anlık olarak sessiz görünüyorsa
            # Kendinden önceki 'hangover_frames' kadarı konuşma mıydı bak
            start_check = max(0, i - hangover_frames)
            if np.any(vad_mask[start_check:i]):
                final_vad_mask[i] = True # Duraksamayı konuşma olarak kabul et
                
    # 5. Voiced / Unvoiced Sınıflandırması
    # Ortalama ZCR ve Enerji eşikleri (daha dinamik de yapılabilir, basit eşikleme kullanıyoruz)
    zcr_threshold = np.mean(zcrs) + 0.5 * np.std(zcrs)
    energy_threshold_vu = np.mean(energies)
    
    is_voiced = np.zeros(num_frames, dtype=bool)
    is_unvoiced = np.zeros(num_frames, dtype=bool)
    
    for i in range(num_frames):
        if final_vad_mask[i]: # Sadece Konuşma (Speech) olan yerleri sınıflandır
            # Voiced (Sesli): Yüksek Enerji, Düşük ZCR
            if energies[i] > (energy_threshold_vu * 0.1) and zcrs[i] < zcr_threshold:
                is_voiced[i] = True
            # Unvoiced (Sessiz): Düşük/Orta Enerji, Yüksek ZCR
            else:
                is_unvoiced[i] = True

    # 6. Sinyal Çıktısı Üretimi (.wav kaydetme)
    # Maskelenmiş frame'leri yeniden inşa ederken OLA (Overlap-Add) yöntemi 
    # en doğrusudur ama basitlik için VAD maskesinden geçen örnekleri uç uca ekliyoruz.
    speech_signal = []
    for i in range(num_frames):
        if final_vad_mask[i]:
            start = i * step_size
            # Örtüşme kısımlarını çift saymamak için sadece step_size kadar kısmı ekliyoruz
            # Son frame ise tamamını ekle
            end = start + step_size if i < num_frames - 1 else start + frame_length
            speech_signal.extend(signal[start:end])
            
    speech_signal = np.array(speech_signal, dtype=signal.dtype)
    wav.write('compressed_speech.wav', sample_rate, speech_signal)
    
    # Süre ve Sıkıştırma Analizi
    orig_duration = len(signal) / sample_rate
    new_duration = len(speech_signal) / sample_rate
    compression_ratio = ((orig_duration - new_duration) / orig_duration) * 100
    
    print(f"Orijinal Sure: {orig_duration:.2f} saniye")
    print(f"Sikistirilmis Sure: {new_duration:.2f} saniye")
    print(f"Elde edilen sikistirma orani: %{compression_ratio:.2f}")

    # 7. Görselleştirme (Matplotlib)
    time_axis = np.linspace(0, orig_duration, len(signal))
    frame_time_axis = np.linspace(0, orig_duration, num_frames)

    plt.figure(figsize=(14, 10))

    # Orijinal Sinyal
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, signal, color='blue', alpha=0.7)
    plt.title("1. Orijinal Ses Sinyali")
    plt.xlabel("Zaman (s)")
    plt.ylabel("Genlik")

    # Enerji ve ZCR (Aynı grafikte iki y ekseni ile)
    ax1 = plt.subplot(3, 1, 2)
    ax1.plot(frame_time_axis, energies, color='orange', label='Karesel Enerji')
    ax1.set_ylabel("Enerji", color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')
    
    ax2 = ax1.twinx()
    ax2.plot(frame_time_axis, zcrs, color='purple', label='Sıfır Geçiş Oranı (ZCR)')
    ax2.set_ylabel("ZCR", color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    plt.title("2. Pencere Bazlı Enerji ve ZCR Grafikleri")

    # Renkli Maskelerle VAD ve Voiced/Unvoiced
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, signal, color='gray', alpha=0.5)
    
    # Arka plana maskeleri çizelim
    for i in range(num_frames):
        start_t = (i * step_size) / sample_rate
        end_t = (start_t + (frame_length_ms / 1000))
        
        if is_voiced[i]:
            plt.axvspan(start_t, end_t, color='green', alpha=0.3, lw=0)
        elif is_unvoiced[i]:
            plt.axvspan(start_t, end_t, color='yellow', alpha=0.3, lw=0)
            
    # Sadece label (lejant) gösterebilmek için görünmez çizgiler
    plt.axvspan(-1, -1, color='green', alpha=0.3, label='Sesli Harfler (Voiced)')
    plt.axvspan(-1, -1, color='yellow', alpha=0.3, label='Sessiz Harfler (Unvoiced)')
    
    plt.title("3. Zaman Domeninde Voiced (Yeşil) / Unvoiced (Sarı) Bölgeleri")
    plt.xlabel("Zaman (s)")
    plt.xlim(0, orig_duration)
    plt.legend()

    plt.tight_layout()
    # Ekrana çizdirmek için kodları kendi bilgisayarında çalıştırdığında aşağıdaki yorumu kaldırabilirsin:
    # plt.show()
    plt.savefig('analiz_sonucu.png')
    print("Gorseller 'analiz_sonucu.png' olarak kaydedildi.")

# Fonksiyonu çağırma örneği:
dosya_yolu = "audio.wav"  # Kendi ses dosyanızın adını buraya yazın
process_audio(dosya_yolu)