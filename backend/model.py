import os
import numpy as np
import onnxruntime as ort

from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "dr_model.onnx"
)


# ============================================================
# DR CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def preprocess_image(image):
    """
    Convert PIL image into the exact format
    expected by the EfficientNet-B0 ONNX model.
    """

    # Make sure image is RGB
    image = image.convert("RGB")

    # Resize exactly like torchvision
    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    # PIL → NumPy
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Convert 0-255 → 0-1
    image_array = image_array / 255.0

    # Normalize using ImageNet mean/std
    image_array = (
        image_array - MEAN
    ) / STD

    # HWC → CHW
    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array.astype(
        np.float32
    )


# ============================================================
# LOAD ONNX MODEL
# ============================================================

def load_model():

    print("Loading DR model...")

    print(
        f"ONNX model path: {MODEL_PATH}"
    )

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"ONNX model file not found: {MODEL_PATH}"
        )

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    print(
        "ONNX model loaded successfully."
    )

    return session


# ============================================================
# LOAD MODEL ONCE
# ============================================================

model = load_model()

INPUT_NAME = model.get_inputs()[0].name
OUTPUT_NAME = model.get_outputs()[0].name


# ============================================================
# SOFTMAX
# ============================================================

def softmax(logits):

    logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True
    )

    exp_values = np.exp(logits)

    return exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )


# ============================================================
# PREDICT SINGLE IMAGE
# ============================================================

def predict_image(image):
    """
    Predict diabetic retinopathy grade
    for one PIL image.

    Returns:
        predicted_class
        confidence
        probabilities
    """

    # Preprocess image
    input_array = preprocess_image(
        image
    )

    # ONNX prediction
    outputs = model.run(
        [OUTPUT_NAME],
        {
            INPUT_NAME: input_array
        }
    )

    logits = outputs[0]

    # Convert logits → probabilities
    probabilities = softmax(
        logits
    )[0]

    # Highest probability class
    predicted_class = int(
        np.argmax(probabilities)
    )

    # Confidence
    confidence = float(
        probabilities[predicted_class]
    )

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# PREDICT BATCH OF IMAGES
# ============================================================

def predict_batch(images):
    """
    Predict multiple images.

    Used by LIME because LIME generates
    many modified versions of the image.
    """

    batch = []

    for image in images:

        # Keep values between 0 and 1
        image = np.clip(
            image,
            0,
            1
        )

        # NumPy → PIL
        image_pil = Image.fromarray(
            (
                image * 255
            ).astype(
                np.uint8
            )
        ).convert("RGB")

        # Preprocess
        processed = preprocess_image(
            image_pil
        )[0]

        batch.append(
            processed
        )

    # Combine into batch
    batch = np.stack(
        batch
    ).astype(
        np.float32
    )

    # ONNX prediction
    outputs = model.run(
        [OUTPUT_NAME],
        {
            INPUT_NAME: batch
        }
    )

    logits = outputs[0]

    # Convert logits → probabilities
    probabilities = softmax(
        logits
    )

    return probabilities