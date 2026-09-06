import os
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from backend.model import predict_image, CLASS_NAMES
from backend.lime_explainer import explain_image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

LIME_DIR = os.path.join(BASE_DIR, "lime_results")

os.makedirs(LIME_DIR, exist_ok=True)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Explainable Diabetic Retinopathy AI",
    description="Diabetic Retinopathy prediction with LIME explainability",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def serve_frontend():

    index_path = os.path.join(
        FRONTEND_DIR,
        "index.html"
    )

    if not os.path.isfile(index_path):
        raise HTTPException(
            status_code=404,
            detail="Frontend index.html not found."
        )

    return FileResponse(
        index_path,
        media_type="text/html"
    )


# ============================================================
# CSS
# ============================================================

@app.get("/style.css")
def serve_css():

    css_path = os.path.join(
        FRONTEND_DIR,
        "style.css"
    )

    if not os.path.isfile(css_path):
        raise HTTPException(
            status_code=404,
            detail="style.css not found."
        )

    return FileResponse(
        css_path,
        media_type="text/css"
    )


# ============================================================
# JAVASCRIPT
# ============================================================

@app.get("/script.js")
def serve_javascript():

    js_path = os.path.join(
        FRONTEND_DIR,
        "script.js"
    )

    if not os.path.isfile(js_path):
        raise HTTPException(
            status_code=404,
            detail="script.js not found."
        )

    return FileResponse(
        js_path,
        media_type="application/javascript"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "loaded",
        "lime": "available"
    }


# ============================================================
# PREDICT + LIME
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Check file type
    # --------------------------------------------------------

    if not file.content_type or not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        contents = await file.read()

        image = Image.open(
            __import__("io").BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    try:

        predicted_class, confidence, probabilities = predict_image(
            image
        )

        predicted_class = int(predicted_class)

        confidence = float(confidence)

        probabilities = [
            float(p)
            for p in probabilities
        ]

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {str(e)}"
        )


    # --------------------------------------------------------
    # LIME EXPLANATION
    # --------------------------------------------------------

    lime_filename = None

    lime_class = predicted_class

    lime_confidence = confidence

    try:

        lime_image, lime_class, lime_confidence = explain_image(
            image,
            num_samples=100
        )

        # Convert NumPy values to Python values
        lime_class = int(lime_class)

        lime_confidence = float(lime_confidence)


        # Unique filename
        lime_filename = (
            f"lime_{uuid.uuid4().hex}.jpg"
        )


        lime_path = os.path.join(
            LIME_DIR,
            lime_filename
        )


        # Save LIME result
        lime_image.save(
            lime_path,
            format="JPEG",
            quality=95
        )

    except Exception as e:

        print("LIME ERROR:", e)

        lime_filename = None


    # ========================================================
    # RESPONSE
    # ========================================================

    response = {

        "success": True,

        "filename": file.filename,

        "prediction": {

            "predicted_class": int(
                predicted_class
            ),

            "class_name": CLASS_NAMES[
                int(predicted_class)
            ],

            "confidence": round(
                float(confidence) * 100,
                2
            ),

            "probabilities": {

                CLASS_NAMES[i]: round(
                    float(probabilities[i]) * 100,
                    2
                )

                for i in range(
                    len(CLASS_NAMES)
                )
            }
        },

        "lime": {

            "available": (
                lime_filename is not None
            ),

            "predicted_class": int(
                lime_class
            ),

            "class_name": CLASS_NAMES[
                int(lime_class)
            ],

            "confidence": round(
                float(lime_confidence) * 100,
                2
            ),

            "image_url": (

                f"/lime/{lime_filename}"

                if lime_filename

                else None
            )
        }
    }


    return response


# ============================================================
# SERVE LIME IMAGE
# ============================================================

@app.get("/lime/{filename}")
def get_lime_image(filename: str):

    file_path = os.path.join(
        LIME_DIR,
        filename
    )


    if not os.path.isfile(file_path):

        raise HTTPException(
            status_code=404,
            detail="LIME image not found."
        )


    return FileResponse(
        file_path,
        media_type="image/jpeg"
    )