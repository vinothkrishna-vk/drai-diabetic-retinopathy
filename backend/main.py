import os
import gc
import io
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from backend.model import predict_image, CLASS_NAMES
from backend.lime_explainer import explain_image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)


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
    # CHECK FILE TYPE
    # --------------------------------------------------------

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    contents = None
    image = None

    try:

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )

    finally:

        contents = None

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    try:

        predicted_class, confidence, probabilities = predict_image(
            image
        )

        predicted_class = int(
            predicted_class
        )

        confidence = float(
            confidence
        )

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
    # DR GRADE
    # --------------------------------------------------------

    # The model uses the standard 5 DR grades:
    #
    # 0 = No DR
    # 1 = Mild
    # 2 = Moderate
    # 3 = Severe
    # 4 = Proliferative

    dr_grade = predicted_class

    dr_grade_label = f"Grade {dr_grade}"

    # --------------------------------------------------------
    # LIME EXPLANATION
    # --------------------------------------------------------

    lime_available = False

    lime_class = predicted_class

    lime_confidence = confidence

    lime_image_data = None

    try:

        print(
            "Starting lightweight LIME explanation..."
        )

        lime_image, lime_class, lime_confidence = explain_image(
            image,
            num_samples=25
        )

        lime_class = int(
            lime_class
        )

        lime_confidence = float(
            lime_confidence
        )

        # ----------------------------------------------------
        # Convert LIME image to JPEG in memory
        # ----------------------------------------------------

        image_buffer = io.BytesIO()

        lime_image.save(
            image_buffer,
            format="JPEG",
            quality=85,
            optimize=True
        )

        image_bytes = image_buffer.getvalue()

        # ----------------------------------------------------
        # Convert JPEG to Base64
        # ----------------------------------------------------

        encoded_image = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        lime_image_data = (
            "data:image/jpeg;base64,"
            + encoded_image
        )

        lime_available = True

        print(
            "LIME explanation generated successfully."
        )

        print(
            f"LIME image size: "
            f"{len(image_bytes) / 1024:.2f} KB"
        )

        # ----------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------

        del lime_image
        del image_buffer
        del image_bytes
        del encoded_image

    except Exception as e:

        print(
            "LIME ERROR:",
            str(e)
        )

        lime_available = False

        lime_image_data = None

    finally:

        gc.collect()

    # ========================================================
    # RESPONSE
    # ========================================================

    response = {

        "success": True,

        "filename": file.filename,

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        "prediction": {

            # Numerical DR grade
            "grade": dr_grade,

            # Human-readable grade
            "grade_label": dr_grade_label,

            # Existing predicted class
            "predicted_class": predicted_class,

            # Existing class name
            "class_name": CLASS_NAMES[
                predicted_class
            ],

            # Confidence
            "confidence": round(
                confidence * 100,
                2
            ),

            # All class probabilities
            "probabilities": {

                CLASS_NAMES[i]: round(
                    probabilities[i] * 100,
                    2
                )

                for i in range(
                    len(CLASS_NAMES)
                )
            }
        },

        # ----------------------------------------------------
        # LIME
        # ----------------------------------------------------

        "lime": {

            "available": lime_available,

            "grade": lime_class,

            "grade_label": f"Grade {lime_class}",

            "predicted_class": lime_class,

            "class_name": CLASS_NAMES[
                lime_class
            ],

            "confidence": round(
                lime_confidence * 100,
                2
            ),

            # Direct image data for browser
            "image_data": lime_image_data
        }
    }

    # --------------------------------------------------------
    # FINAL MEMORY CLEANUP
    # --------------------------------------------------------

    gc.collect()

    return response