# Python Server (`server.py`)

## Overview
Simple FastAPI service that detects pool balls in an image and returns circle positions with labels:
- `cue`
- `object (solid)`
- `object (stripe)`

## Tech Stack
- Python 3.10+  
- FastAPI + Uvicorn  
- OpenCV + NumPy + Pillow  
- CORS enabled for local/dev testing

## Setup

### 1) Clone and enter the folder
```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd "Python server"
```

### 2) Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install manually:
```bash
pip install fastapi uvicorn opencv-python numpy pillow
```

## Run
```bash
python server.py
# Uvicorn running on http://0.0.0.0:8000
```

## API

### POST `/detect`
- **Body**: `multipart/form-data` with key `file` as an image (JPG/PNG).
- **Response**:
```json
{
  "success": true,
  "detections": [
    {"x": 123, "y": 456, "r": 18, "label": "cue"},
    {"x": 234, "y": 345, "r": 17, "label": "object (solid)"},
    {"x": 345, "y": 222, "r": 16, "label": "object (stripe)"}
  ],
  "w": 1920,
  "h": 1080
}
```
- `x, y` are circle centers in original image pixels, `r` is radius in pixels, `w/h` are image width/height.

### Test with `curl`
```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@/full/path/to/your/pool_image.jpg"
```

### Test in Postman
- Method: `POST`
- URL: `http://localhost:8000/detect`
- Body: `form-data` → key `file` (type: File)

## Notes & Tips
- CORS is open by default for development (`allow_origins=["*"]`).
- The server performs circle detection and simple color/stripe classification; it is not a learned model. Lighting/occlusion can affect results.
- For best results: reasonably clear table area, balls not heavily occluded, moderate shadows.

## Deploy (basic)
- Any Python host (Railway, Render, Fly.io, EC2, etc.)
- Use `uvicorn` in a process manager:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```
- Remember to install system packages required by OpenCV if your platform needs them.

## Troubleshooting
- **OpenCV import error**: ensure the venv is activated before installing/running.
- **500 errors**: check the server logs; confirm the uploaded file is a valid image.
- **CORS issues on device**: if running from a phone, use your machine’s LAN IP instead of `localhost` and allow that origin.
