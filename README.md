# 🧠 Deep Live Cam — Real-Time Face Swap by Sheva Ramdhani

> 🎥 Real-time AI-powered face tracking and swapping system using Python, Deep Learning, and GPU acceleration.

This project allows **real-time facial recognition and swapping** using your webcam. Optimized for **CUDA GPU**, **ONNX Runtime**, and includes **FFmpeg** for smooth video handling.

---

## ✨ Features

* 🔍 Real-time webcam capture
* 🧠 Deep Learning face detection and swapping
* ⚡ GPU acceleration (CUDA / TensorRT)
* 🎨 Optional face enhancement
* 🪞 Live mirror UI with CustomTkinter
* 🧩 Modular architecture: Torch, ONNXRuntime, TensorFlow backend

---

## 🧰 Dependencies

### Python

* Python 3.12.x
* Virtual environment recommended:

```bash
python -m venv venv
venv\Scripts\activate
```

### Python Libraries

```bash
pip install --upgrade pip setuptools wheel

pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128
pip install numpy>=1.23.5,<2
pip install typing-extensions>=4.8.0
pip install opencv-python==4.10.0.84
pip install cv2_enumerate_cameras==1.1.15
pip install onnx==1.18.0
pip install insightface==0.7.3
pip install psutil==5.9.8
pip install pillow==11.1.0
pip install customtkinter==5.2.2
pip install tensorflow
pip install protobuf==4.25.1
pip install onnxruntime-gpu==1.22.0
```

---

## 🎬 FFmpeg Setup

* FFmpeg executables included (`ffmpeg.exe`, `ffplay.exe`, `ffprobe.exe`)
* Add FFmpeg to system `PATH` if not already:

```
C:\Users\ramdh\Documents\DeepLearn\Deep-Live-Cam
```

* Verify installation:

```bash
ffmpeg -version
```

---

## ⚡ CUDA & cuDNN

### Compatible Versions

* CUDA Toolkit: 12.8
* cuDNN: 9.4
* GPU: NVIDIA RTX 20xx / 30xx / 40xx or newer

### Setup

1. Install CUDA 12.8
2. Install cuDNN 9.4 for CUDA 12
3. Copy cuDNN files to CUDA directory (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\`)
4. Add to `PATH`:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp
```

5. Verify GPU in Python:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## 🧩 ONNX Runtime GPU

Check available providers:

```python
import onnxruntime as ort
print("Available:", ort.get_available_providers())

session = ort.InferenceSession(
    r"models\inswapper_128_fp16.onnx",
    providers=["CUDAExecutionProvider"]
)
print("Active:", session.get_providers())
```

✅ Example output:

```
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## 🚀 Running the Project

Run main script with GPU support:

```bash
python run.py
```

Alternatively, use batch scripts:

```bash
run-cuda.bat      # Force CUDA GPU execution
run-directml.bat  # DirectML execution (Windows GPU API)
```

---

## 🏗️ Project Structure

```
Deep-Live-Cam/
├── models/
│   ├── inswapper_128.onnx
│   └── inswapper_128_fp16.onnx
├── modules/
│   ├── core.py
│   ├── ui.py
│   └── ...
├── CUDA.py
├── ffmpeg.exe
├── ffplay.exe
├── ffprobe.exe
├── run.py
├── run-cuda.bat
├── run-directml.bat
├── requirements.txt
├── README.md
└── ...
```

---

## 🛠️ Troubleshooting

* GPU not detected: ensure **CUDA 12.8** and **cuDNN 9.4** installed and PATH updated.
* Remove old CUDA 13 leftovers if present.
* Restart PC after updating environment variables.
* If ONNX fails to load model, check `models/` folder paths.

---

## 👨‍💻 Author

**Sheva Ramdhani**
Deep Learning & Computer Vision Enthusiast

---

## 📜 License

[MIT License](LICENSE)
