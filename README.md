# Computer Vision & AI Environment — Household Face Recognition

Welcome to the **Computer Vision & AI** project! This repository contains a production-structured Python Computer Vision system supporting real-time person tracking, crop recording, Person Re-ID, and local Household Face Recognition.

---

## 🎯 Project Roadmap

The long-term goal is to build an industry-grade Computer Vision system:
```text
CCTV / Laptop Webcam → Video Stream → Pretrained YOLO → Person Detection → Object Tracking → Unique Track ID → Face Detection (YuNet) → Face Alignment (5-Point) → Face Embedding (SFace 128-d) → SQLite Memory Match → Green/Red Bounding Box UI → Log Audit
```

- **Phase 1 (Completed)**: Environment verification & live webcam detection.
- **Phase 2 (Completed)**: Real-time person tracking with temporary Track IDs (ByteTrack).
- **Phase 3 (Completed)**: Person crop extraction, individual session records, and metadata storage.
- **Phase 4 (Completed)**: Person Re-Identification (Re-ID) using OSNet + Hungarian Global Association.
- **Phase 5 (Completed)**: Household Face Recognition using YuNet (5-Point Landmarks), SFace (128-d Embeddings), SQLite Database (`household_ai.db`), Interactive Member Enrollment, and Green/Red UI Overlays.

---

## 🏗️ Project Architecture (Phase 5)

```text
computer-vision-ai/
├── models/
│   ├── yolov8n.pt                           # YOLOv8 Person Detector
│   ├── face_detection_yunet_2023mar.onnx    # YuNet 5-Landmark Face Detector (~230 KB)
│   └── face_recognition_sface_2021dec.onnx  # SFace 128-d Face Feature Extractor (~1.2 MB)
│
├── outputs/
│   ├── database/
│   │   └── household_ai.db                  # Local SQLite database (Persons, Embeddings, Logs)
│   └── enrolled_faces/
│
├── src/
│   ├── config/
│   │   └── settings.py                      # Centralized configuration & thresholds
│   ├── detection/
│   │   ├── yolo_detector.py                 # YOLO Person Detector wrapper
│   │   └── face_detector.py                 # YuNet 5-landmark face detector wrapper
│   ├── tracking/
│   │   └── tracker.py                       # ByteTrack tracking wrapper
│   ├── recognition/
│   │   ├── face_quality.py                  # Blur (Laplacian), Size, Lighting checks
│   │   ├── face_alignment.py                # 5-point landmark affine face aligner
│   │   ├── face_embedder.py                 # SFace 128-d L2-normalized embedder
│   │   └── face_matcher.py                  # Cosine similarity matcher & candidate ranker
│   ├── storage/
│   │   ├── database.py                      # SQLite connection & schema initialization
│   │   ├── person_repository.py             # Persons CRUD operations (UUID identity)
│   │   ├── embedding_repository.py          # NumPy binary blob vector serialization
│   │   └── log_repository.py                # Historical recognition audit logger
│   ├── enrollment/
│   │   ├── enrollment_manager.py            # Sample capture & validation manager
│   │   └── enroll_person.py                 # Interactive CLI & camera enrollment app
│   ├── ui/
│   │   └── overlay.py                       # Green (Known) / Red (Unknown) UI renderer
│   ├── phase5_face_recognition.py           # Main Phase 5 Real-Time Application
│   └── test/                                # Automated Unit Test Suite
│       ├── test_embedding.py
│       ├── test_similarity.py
│       ├── test_face_quality.py
│       └── test_matching.py
│
├── docs/
│   └── PHASE-5.md                           # Comprehensive Hinglish educational guide
├── run_phase5.bat                           # Double-click Windows launcher for Phase 5
└── run_enrollment.bat                       # Double-click Windows launcher for Enrollment
```

---

## 📚 Phase 5 Concepts Summary

| Concept | Simple Hinglish Meaning | Technical Definition |
| :--- | :--- | :--- |
| **Face Detection** | *"Face kahan hai?"* | YuNet ONNX detector returning bounding box `[x, y, w, h]` and 5 facial landmarks. |
| **Face Alignment** | *"Face ko seedha karna"* | 5-point affine transformation aligning eyes horizontally to `(112, 112)` resolution. |
| **Face Embedding** | *"Face ka digital passport"* | 128-dimensional L2-normalized float vector extracted by SFace neural network. |
| **Cosine Similarity** | *"Vectors kitne milte-julte hain?"* | Dot product of unit vectors ($\mathbf{u} \cdot \mathbf{v}$) ranging from `-1.0` to `1.0`. |
| **Enrollment** | *"Naye member ke embeddings save karna"* | Capturing 10 reference face samples and storing vectors in SQLite database. |
| **Green Bounding Box** | 🟢 **Known Member** | Displayed when similarity $\ge 0.65$ (`Rahul \| Known \| 93%`). |
| **Red Bounding Box** | 🔴 **Unknown Person** | Displayed when candidate similarity $< 0.65$ (`UNKNOWN \| 58%`). |

---

## 🚀 How to Run Phase 5

### 1. Activate Virtual Environment
```powershell
.venv\Scripts\Activate.ps1
```

### 2. Enroll a Household Member (Interactive CLI)
```powershell
python src/enrollment/enroll_person.py
```
*or double-click `run_enrollment.bat`*

### 3. Run Phase 5 Real-Time Face Recognition
```powershell
python src/phase5_face_recognition.py
```
*or double-click `run_phase5.bat`*

### 4. Run Automated Test Suite
```powershell
python -m unittest discover -s src/test -p "test_*.py"
```
