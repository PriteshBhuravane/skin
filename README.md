# SkinFusion Analyzer

AI-powered dermatology assistant that classifies dermatoscopic skin lesion images using an ensemble of ConvNeXt, EfficientNet and ResNet models.

## Repository layout
- backend/  
  - `app.py` — API server (serves `/predict`)  
  - `model.py`, `fix_model.py`, `test_model.py` — model utilities and tests  
  - `best_skfusion.pth` — trained model weights (required by backend)  
  - `requirements.txt` — Python dependencies
- frontend/  
  - `src/App.jsx` — React UI (Vite)  
  - `src/main.jsx`, `index.html`, and other frontend assets  
  - `package.json` — frontend dependencies and scripts

## Prerequisites
- Python 3.8+ and pip
- Node.js 16+ (recommended 18+) and npm or yarn
- Git (optional)

## Backend — run locally
1. Open a terminal and change to the backend folder:
   cd backend
2. Create a virtual environment (recommended) and install dependencies:
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows
   pip install -r requirements.txt
3. Ensure `best_skfusion.pth` is present in the backend folder.
4. Start the API server (example):
   python app.py
   The server is expected to listen on port 8000 and expose POST /predict

Notes:
- The frontend expects the backend at http://localhost:8000
- Uploaded image size limit enforced by the frontend is 10MB

## Frontend — run locally
1. Open a terminal and change to the frontend folder:
   cd frontend
2. Install dependencies:
   npm install
   # or
   yarn
3. Start the dev server:
   npm run dev
4. Open the app in your browser (Vite will show the local URL, commonly http://localhost:5173)

## Usage
- Open the frontend in your browser.
- Upload a dermatoscopic image (PNG/JPG/JPEG, ≤ 10MB).
- Click "Analyze Lesion" to send the image to the backend `/predict` endpoint.
- Results include predicted class, confidence, severity, and a probability distribution.

## Troubleshooting
- If analysis fails, verify the backend process is running on port 8000.
- Check backend logs for model loading errors (missing `best_skfusion.pth` or dependency issues).
- Ensure CORS is enabled in the backend if running frontend and backend on different hosts/ports.

## Tests
- Backend contains `test_model.py` for model-level testing. Run with the virtualenv active:
  python test_model.py

## Notes & Disclaimer
- This tool is for educational purposes only. Not a replacement for professional medical diagnosis.
- Do not use outputs for clinical decision-making without a qualified dermatologist.
