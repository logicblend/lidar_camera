import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

// Ana giriş noktası
List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Kameralar alınamadı: $e');
  }
  runApp(const DollyZoomApp());
}

class DollyZoomApp extends StatelessWidget {
  const DollyZoomApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Dolly Zoom (Vertigo)',
      theme: ThemeData.dark(),
      home: const DollyZoomScreen(),
    );
  }
}

class DollyZoomScreen extends StatefulWidget {
  const DollyZoomScreen({super.key});

  @override
  State<DollyZoomScreen> createState() => _DollyZoomScreenState();
}

class _DollyZoomScreenState extends State<DollyZoomScreen> {
  CameraController? _cameraController;
  
  // Yüz algılayıcı nesnemiz (Hızlı modda, sadece bounding box için tracking açık)
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: false,
      enableLandmarks: false,
      enableClassification: false,
      enableTracking: true,
      performanceMode: FaceDetectorMode.fast,
    ),
  );

  bool _isDetecting = false;
  int _frameCount = 0;
  final int _frameSkip = 3; // Performans için her 3 karede 1 yüz taraması

  // 1. Kalibrasyon ve Başlangıç Değişkenleri
  bool _isLocked = false;
  double _initialFaceWidth = 0.0;
  double _initialZoom = 1.0;

  // Zoom hesaplama ve LPF değişkenleri
  double _targetZoom = 1.0;
  double _currentZoom = 1.0;
  final double _alpha = 0.15; // LPF Yumuşatma katsayısı (ne kadar küçükse o kadar yumuşak)
  double _maxZoom = 1.0;
  double _minZoom = 1.0;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    if (cameras.isEmpty) return;

    // Arka kamerayı seçiyoruz (Dolly Zoom genelde arka kamera ile çekilir)
    final camera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    _cameraController = CameraController(
      camera,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: Platform.isIOS ? ImageFormatGroup.bgra8888 : ImageFormatGroup.yuv420,
    );

    await _cameraController!.initialize();
    
    // Cihazın desteklediği min ve max zoom değerlerini alıyoruz
    _maxZoom = await _cameraController!.getMaxZoomLevel();
    _minZoom = await _cameraController!.getMinZoomLevel();
    _currentZoom = _minZoom;
    _targetZoom = _minZoom;

    // Görüntü akışını başlatıyoruz
    _cameraController!.startImageStream(_processCameraImage);

    if (mounted) setState(() {});
  }

  void _processCameraImage(CameraImage image) async {
    if (!mounted) return;
    
    _frameCount++;

    // --- 3. Titreme Önleyici Yumuşatma (Smoothing / LPF) ---
    // Bu blok, yüz taraması atlansa bile HER karede çalışır.
    // Bu sayede zoom geçişi, _alpha oranıyla yumuşatılarak kesintisiz hissettirir.
    if (_isLocked && (_targetZoom - _currentZoom).abs() > 0.005) {
      _currentZoom = _currentZoom + _alpha * (_targetZoom - _currentZoom);
      _currentZoom = _currentZoom.clamp(_minZoom, _maxZoom);
      
      try {
        await _cameraController!.setZoomLevel(_currentZoom);
      } catch (e) {
        debugPrint('Zoom ayarlama hatası: $e');
      }
    }

    // --- 4. Performans Optimizasyonu (Frame Skipping) ---
    // Yüz tanıma işlemi ağır olduğu için, sadece belirlenen aralıklarda (örn: her 3 karede bir) çalıştır.
    if (_frameCount % _frameSkip != 0) return;
    if (_isDetecting) return; // Hala önceki kare işleniyorsa, yenisini atla

    _isDetecting = true;
    
    try {
      final inputImage = _createInputImage(image);
      if (inputImage == null) return;

      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isNotEmpty) {
        // En net ve ortadaki yüzü kabul ettiğimizi varsayarak ilk yüzü alıyoruz
        final face = faces.first;
        final currentFaceWidth = face.boundingBox.width;

        if (_isLocked && currentFaceWidth > 0) {
          // --- 2. Görüntü İşleme ve Zoom Formülü ---
          // Hedefi aynı boyutta tutmak için formül: Z_hedef = Z_ilk * (W_ilk / W_guncel)
          double calculatedTargetZoom = _initialZoom * (_initialFaceWidth / currentFaceWidth);
          _targetZoom = calculatedTargetZoom.clamp(_minZoom, _maxZoom);
        } else if (!_isLocked) {
          // Kullanıcı henüz "Kilitlen" butonuna basmadıysa,
          // arka planda en güncel yüz genişliğini ve zoom'u kalibrasyon için referans olarak tutuyoruz.
          _initialFaceWidth = currentFaceWidth;
          _initialZoom = _currentZoom;
        }
      }
    } catch (e) {
      debugPrint('Yüz tanıma hatası: $e');
    } finally {
      _isDetecting = false;
    }
  }

  // CameraImage nesnesini ML Kit'in anlayacağı InputImage nesnesine dönüştürme fonksiyonu
  InputImage? _createInputImage(CameraImage image) {
    if (_cameraController == null) return null;
    
    final camera = _cameraController!.description;
    final sensorOrientation = camera.sensorOrientation;
    
    InputImageRotation? rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
    if (rotation == null) return null;

    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    
    // iOS BGRA8888 desteklediği için ona uygun format ayarı
    if (format == null || (Platform.isAndroid && format != InputImageFormat.nv21)) {
      // Çoğu durumda iOS'da çalışacağımız için bu bloğu geçiyoruz.
    }

    final plane = image.planes.first;
    
    return InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation, // Kameranın fiziksel dönüş açısı
        format: Platform.isIOS ? InputImageFormat.bgra8888 : InputImageFormat.nv21,
        bytesPerRow: plane.bytesPerRow,
      ),
    );
  }

  void _toggleLock() {
    setState(() {
      _isLocked = !_isLocked;
      // "Kilitlen" dediğimizde formül devreye girer.
      // _initialFaceWidth ve _initialZoom _processCameraImage içinde arka planda zaten güncellenmişti.
    });
  }

  @override
  void dispose() {
    _cameraController?.stopImageStream();
    _cameraController?.dispose();
    _faceDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(_cameraController!),
          
          // Arayüz - Kontrol Butonu
          Positioned(
            bottom: 50,
            left: 0,
            right: 0,
            child: Column(
              children: [
                Text(
                  _isLocked ? 'HEDEF KİLİTLİ: DOLLY ZOOM AKTİF' : 'HEDEFİ ORTALA VE KİLİTLEN',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    backgroundColor: Colors.black54,
                  ),
                ),
                const SizedBox(height: 20),
                FloatingActionButton(
                  backgroundColor: _isLocked ? Colors.red : Colors.green,
                  onPressed: _toggleLock,
                  child: Icon(
                    _isLocked ? Icons.stop : Icons.lock_outline,
                    size: 32,
                    color: Colors.white,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
