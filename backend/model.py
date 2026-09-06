import os
import numpy as np
import torch

from PIL import Image
from torch import nn
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

# Get the project root:
# dr_web_app/
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Path to trained model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "dr_model_best.pth"
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
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading DR model...")

    print(f"Model path: {MODEL_PATH}")

    print(f"Using device: {DEVICE}")

    # Create EfficientNet-B0
    model = models.efficientnet_b0(
        weights=None
    )

    # Change final classifier to 5 classes
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        5
    )

    # Check model file
    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    # Load trained weights
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    # Move model to CPU/GPU
    model = model.to(DEVICE)

    # Evaluation mode
    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# LOAD MODEL ONCE
# ============================================================

model = load_model()


# ============================================================
# PREDICT SINGLE IMAGE
# ============================================================

def predict_image(image):
    """
    Predict diabetic retinopathy grade for one PIL image.

    Returns:
        predicted_class
        confidence
        probabilities
    """

    # Make sure image is RGB
    image = image.convert("RGB")

    # Apply preprocessing
    input_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    # Disable gradient calculation
    with torch.no_grad():

        # Model prediction
        outputs = model(
            input_tensor
        )

        # Convert logits to probabilities
        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Get highest probability class
        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    # Get confidence
    confidence = probabilities[
        0,
        predicted_class
    ].item()

    # Convert probabilities to NumPy
    probabilities = probabilities[
        0
    ].cpu().numpy()

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

    Used later by LIME because LIME generates
    many modified versions of the image.
    """

    batch = []

    for image in images:

        # Convert NumPy image to valid range
        image = np.clip(
            image,
            0,
            1
        )

        # Convert NumPy → PIL
        image_pil = Image.fromarray(
            (
                image * 255
            ).astype(
                np.uint8
            )
        ).convert("RGB")

        # Apply preprocessing
        tensor = transform(
            image_pil
        )

        batch.append(
            tensor
        )

    # Combine images into one batch
    batch = torch.stack(
        batch
    ).to(DEVICE)

    # Model prediction
    with torch.no_grad():

        outputs = model(
            batch
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    return probabilities.cpu().numpy()