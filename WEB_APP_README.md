# Museo Scholar Web App

## Installation

```powershell
pip install flask google-generativeai pillow ultralytics opencv-python
```

## Setup

1. **Set your Gemini API key:**

```powershell
$env:GEMINI_API_KEY = "your-api-key-here"
```

Get your free API key from: https://makersuite.google.com/app/apikey

2. **Run the app:**

```powershell
python webcam_web_app.py
```

3. **Open your browser:**
   Navigate to `http://localhost:5000`

## How to Use

1. The webcam feed will appear on the left with YOLO detections
2. Click on any detected object (inside the bounding box)
3. The right panel will show:
   - Object classification
   - Detection confidence
   - AI-generated detailed description from Gemini 1.5 Flash

## Features

- 🎥 Real-time webcam object detection
- 🖱️ Click-to-select objects
- 🤖 Google Gemini 1.5 Flash multimodal analysis
- 📊 Confidence scores with visual bars
- 🎨 Beautiful, responsive UI
- 🏛️ Optimized for museum artifacts (Golden Tara, Manunggul Jar, etc.)

## Notes

- Requires Google Gemini API key (free tier available)
- Using `gemini-1.5-flash` model for fast, cost-effective multimodal analysis
- Click coordinates are mapped to bounding boxes automatically
- AI provides cultural/historical context for detected artifacts
