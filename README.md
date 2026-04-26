# 🎭 Virtual Wig Try-On  
### 6th Semester Computer Vision Project  
**Developed by Safwan Kasi**

---

## 📌 Overview
A real-time **Computer Vision application** that allows users to try different hairstyles using a webcam.  
Ideal for **barbershops and personal styling**, this system provides a live preview before making real-life changes.

---

## ✨ Features
- 🎥 Real-time face detection using Haar Cascades (Classical Computer Vision)
- 👀 Eyes & nose detection for accurate wig alignment
- 👥 Supports multiple faces per frame
- 🎯 Automatic wig positioning and scaling based on:
  - Face size
  - Hair category (men / women / long)
- 🔄 Head rotation tracking (wig stays aligned when head tilts)
- 🔍 Adjustable wig size using keyboard controls (+ / -)
- 🪞 Mirror mode enabled by default
- 🧰 Optional debug overlays (face + feature bounding boxes)
- ⚡ Lightweight — no AI/Deep Learning models required

---

## 🚀 Installation

### 1. Clone Repository

git clone <repository-url>

cd Computer-Vision

### 2. Create Virtual Environment

python -m venv venv

## Activate Environment

### Windows (CMD)

venv\Scripts\activate

### PowerShell

.\venv\Scripts\Activate.ps1

### Linux / macOS

source venv/bin/activate

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Generate Sample Wigs (Optional)

python generate_sample_wigs.py

## ▶️ Usage

python main.py

## 🎮 Controls
### Key	Action

N / →	Next wig

P / ←	Previous wig

+ / -	Increase / decrease wig size

B	Toggle face bounding box

F	Toggle eyes/nose boxes

M	Toggle mirror mode

Q / ESC	Quit

## 💇 Hair Categories

- Wigs are automatically categorized using filename prefixes:

long_ → Long hair (aligned higher on head)

curly_ → Women styles

short_, spiky_, afro_ → Men styles

- Note: Proper naming ensures correct positioning and scaling.

## 🎨 Adding Custom Wigs

### Steps

- Create PNG images with transparent backgrounds
- Place them in:

assets/wigs/

- Restart the application
- Tips for Best Results
- Use front-facing wig images
- Maintain PNG transparency
- Leave space at the bottom (forehead area)
- Recommended size: 400–600 px width

## 📁 Project Structure

Computer-Vision/

├── main.py

├── requirements.txt

├── generate_sample_wigs.py

├── README.md

├── assets/

│   └── wigs/

└── src/

│
    └── __init__.py

│   
    └── wig_overlay.py

## ⚙️ Requirements
- Python 3.8+
- Webcam
- OpenCV
- NumPy

## 🛠️ Troubleshooting
- Camera Not Detected
- Ensure webcam is connected and not used by another app
- Try changing:
camera_index = 1
- Wigs Not Showing
python generate_sample_wigs.py
- Ensure PNG files exist in assets/wigs/
- Poor Tracking
- Ensure good lighting
- Face the camera directly
- Maintain proper distance

## 🎓 Academic Context

- This project was developed as part of a 6th Semester Computer Vision coursework, focusing on:
- Classical Computer Vision techniques
- Real-time image processing
- Human-computer interaction

## 📜 License

MIT License
