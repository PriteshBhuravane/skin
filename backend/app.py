"""
FastAPI Server for SkinFusion-Net
with Gemini image validation as gatekeeper + low-confidence flag.

Flow:
1) User uploads image
2) Gemini checks: "Is this a skin lesion image?" (YES/NO)
3) If YES -> local 7-class HAM10000 model predicts
4) If NO/unsure -> 400 error: "Upload valid skin lesion image"
5) If prediction confidence < MIN_CONFIDENCE -> still return 200,
   but mark as "uncertain" prediction so frontend can show a warning.
"""

import os
import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from model import SkinFusionNet  # your existing model

# -----------------------------------------------------------------------------
# Gemini SDK imports
# -----------------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("skin-fusion-backend")

# -----------------------------------------------------------------------------
# HAM10000 Class Info
# -----------------------------------------------------------------------------
CLASS_INFO = {
    0: {
        "name": "Actinic Keratoses",
        "code": "akiec",
        "description": "Pre-cancerous lesions caused by sun damage",
        "severity": "Medium",
        "color": "#FF9800",
    },
    1: {
        "name": "Basal Cell Carcinoma",
        "code": "bcc",
        "description": "Most common type of skin cancer",
        "severity": "High",
        "color": "#F44336",
    },
    2: {
        "name": "Benign Keratosis",
        "code": "bkl",
        "description": "Non-cancerous skin growth",
        "severity": "Low",
        "color": "#4CAF50",
    },
    3: {
        "name": "Dermatofibroma",
        "code": "df",
        "description": "Common benign skin lesion",
        "severity": "Low",
        "color": "#2196F3",
    },
    4: {
        "name": "Melanoma",
        "code": "mel",
        "description": "Most dangerous form of skin cancer",
        "severity": "Critical",
        "color": "#9C27B0",
    },
    5: {
        "name": "Melanocytic Nevi",
        "code": "nv",
        "description": "Common moles, usually benign",
        "severity": "Low",
        "color": "#8BC34A",
    },
    6: {
        "name": "Vascular Lesions",
        "code": "vasc",
        "description": "Blood vessel-related skin changes",
        "severity": "Low",
        "color": "#00BCD4",
    },
}

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
BACKBONES = [
    "convnext_base.fb_in22k_ft_in1k",
    "tf_efficientnet_b3_ns",
    "resnet50",
]

MODEL_PATH = "best_skfusion.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gemini config
GEMINI_MODEL = "gemini-2.0-flash"  # model name you're using
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# TEMP: hard-code key for local testing if env var not set
# ❗ Put your real key here, but DO NOT commit to GitHub.
if not GEMINI_API_KEY:
    GEMINI_API_KEY = "PASTE_YOUR_GEMINI_KEY"  # <-- replace with real key

# Classification minimum confidence for "high confidence"
MIN_CONFIDENCE = 0.70  # 70%

# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
transform = transforms.Compose(
    [
        transforms.Resize(int(256 / 0.875)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# -----------------------------------------------------------------------------
# Globals: model + Gemini client
# -----------------------------------------------------------------------------
model = None
gemini_client = None


def init_gemini_client():
    """Initialize Gemini client if SDK + API key are available."""
    global gemini_client

    if genai is None:
        logger.warning("google-genai not installed. Gemini validation DISABLED.")
        gemini_client = None
        return

    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.warning("GEMINI_API_KEY / GOOGLE_API_KEY not set. Gemini validation DISABLED.")
        gemini_client = None
        return

    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini client initialized successfully.")
        logger.info(f"   Model for validation: {GEMINI_MODEL}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini client: {e}")
        gemini_client = None


def gemini_is_skin_lesion(image_bytes: bytes, mime_type: str) -> bool:
    """
    Ask Gemini: is this image a human skin lesion / mole / rash suitable for dermatology?

    Returns True  -> allow classification
            False -> reject (invalid / non-skin image)
    """
    if gemini_client is None:
        # If Gemini not available, skip validation (for safety you COULD return False)
        logger.warning("Gemini validation not available. Skipping validation.")
        return True

    prompt = """
You are a strict medical image filter for a skin lesion classifier.

Question: Is this image a close-up of HUMAN SKIN showing a mole, lesion, rash, spot, or similar skin condition,
clearly visible and suitable for dermatology analysis?

Answer RULES:
- Answer with EXACTLY ONE WORD: YES or NO.
- If it is any of these, answer NO:
  - flowers, cars, animals, landscapes, documents
  - full body photos, selfies, faces without zoomed-in lesion
  - X-rays, MRI, CT, histology slides
  - cartoon / drawing / AI generated art
- If you are NOT SURE, answer NO.
"""

    try:
        part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, part],
        )

        text = (response.text or "").strip().upper()
        logger.info(f"Gemini response for skin check: {text!r}")

        if text.startswith("YES"):
            return True
        return False

    except Exception as e:
        logger.error(f"Gemini validation error: {e}")
        # On error, safest is to REJECT
        return False


def load_classification_model():
    """Load 7-class SkinFusionNet model."""
    global model

    logger.info("=" * 80)
    logger.info("LOADING CLASSIFICATION MODEL")
    logger.info("=" * 80)

    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found: {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = SkinFusionNet(BACKBONES, num_classes=7, hidden_dim=1024, dropout=0.3)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    logger.info("✅ Classification model loaded successfully!")
    logger.info(f"   Device: {DEVICE}")
    logger.info(f"   Backbones: {BACKBONES}")
    logger.info("=" * 80)


# -----------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_gemini_client()
    load_classification_model()
    logger.info("SERVER READY")
    yield
    logger.info("Server shutting down...")


app = FastAPI(title="SkinFusion-Net API (Gemini validated)", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Health checks
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "online",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "gemini_validation": gemini_client is not None,
        "gemini_model": GEMINI_MODEL if gemini_client else None,
        "min_confidence": MIN_CONFIDENCE,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "gemini_available": gemini_client is not None,
    }


# -----------------------------------------------------------------------------
# Predict endpoint
# -----------------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict skin lesion class from uploaded image.

    Steps:
    1) Basic file type check
    2) Gemini validation: is it a skin lesion?
    3) If yes -> 7-class classification
    4) Mark prediction as low/high confidence and confident/uncertain
    """

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        logger.info("=" * 80)
        logger.info("📸 NEW PREDICTION REQUEST")
        logger.info(f"   Filename: {file.filename}")
        logger.info(f"   Content type: {file.content_type}")
        logger.info(f"   Image size: {image.size}")
        logger.info("=" * 80)

        # ------------------------------------------------------------------
        # STEP 1: Gemini validation
        # ------------------------------------------------------------------
        logger.info("🔍 STEP 1: Gemini validation - is this a skin lesion?")
        mime_type = file.content_type or "image/jpeg"
        is_skin = gemini_is_skin_lesion(image_bytes, mime_type)

        if not is_skin:
            logger.warning("❌ Gemini: NOT a valid skin lesion image. Rejecting.")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid image type",
                    "message": (
                        "This does not appear to be a valid skin lesion image. "
                        "Please upload a close-up photo of a mole, lesion, or rash."
                    ),
                    "validator": "gemini",
                },
            )

        logger.info("✅ Gemini: Image accepted as skin lesion. Proceeding to classification.")

        # ------------------------------------------------------------------
        # STEP 2: Classification
        # ------------------------------------------------------------------
        logger.info("📊 STEP 2: Running classification.")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = F.softmax(logits, dim=1)[0]

        probs = probabilities.cpu().numpy()
        pred_class = int(probs.argmax())
        confidence = float(probs[pred_class])

        logger.info(
            f"   Predicted class: {pred_class} ({CLASS_INFO[pred_class]['name']})"
        )
        logger.info(f"   Confidence: {confidence * 100:.2f}%")

        # Low-confidence flag + prediction status
        is_low_confidence = confidence < MIN_CONFIDENCE
        confidence_flag = "low" if is_low_confidence else "high"
        prediction_status = "uncertain" if is_low_confidence else "confident"

        if is_low_confidence:
            logger.warning(
                f"⚠️ LOW CONFIDENCE PREDICTION: {confidence * 100:.2f}% "
                f"< {MIN_CONFIDENCE * 100:.2f}% threshold"
            )
        else:
            logger.info(
                f"✅ Confidence above threshold ({MIN_CONFIDENCE * 100:.2f}%)"
            )

        if pred_class not in CLASS_INFO:
            logger.error(f"Invalid predicted class: {pred_class}, defaulting to 'nv'")
            pred_class = 5

        class_info = CLASS_INFO[pred_class]

        # Build response
        all_probs = [
            {
                "class_id": i,
                "class_name": CLASS_INFO[i]["name"],
                "probability": round(float(probs[i]) * 100, 2),
                "color": CLASS_INFO[i]["color"],
            }
            for i in range(len(CLASS_INFO))
        ]
        all_probs.sort(key=lambda x: x["probability"], reverse=True)

        result = {
            "predicted_class": pred_class,
            "class_name": class_info["name"],
            "class_code": class_info["code"],
            "confidence": round(confidence * 100, 2),
            "description": class_info["description"],
            "severity": class_info["severity"],
            "color": class_info["color"],

            # Validation info
            "validator": "gemini",
            "skin_validated": True,

            # New confidence info
            "is_low_confidence": is_low_confidence,
            "confidence_flag": confidence_flag,                 # "low" or "high"
            "confidence_threshold": round(MIN_CONFIDENCE * 100, 2),
            "prediction_status": prediction_status,             # "uncertain" or "confident"

            "all_probabilities": all_probs,
        }

        logger.info("✅ PREDICTION COMPLETE")
        logger.info(f"   Final class: {class_info['name']}")
        logger.info(f"   Severity: {class_info['severity']}")
        logger.info(f"   Prediction status: {prediction_status}")
        logger.info("=" * 80)

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# -----------------------------------------------------------------------------
# Local run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
