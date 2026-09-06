import os
import numpy as np
from PIL import Image

from lime import lime_image
from skimage.segmentation import mark_boundaries

from backend.model import predict_batch, CLASS_NAMES


# ============================================================
# LIME EXPLAINER
# ============================================================

def explain_image(image, num_samples=500):
    """
    Generate a LIME explanation for a retinal image.

    Parameters:
        image: PIL Image
        num_samples: Number of LIME perturbations

    Returns:
        explanation_image: PIL Image
        predicted_class: int
        confidence: float
    """

    # --------------------------------------------------------
    # Convert image to RGB
    # --------------------------------------------------------

    image = image.convert("RGB")

    # LIME works with NumPy arrays
    image_np = np.array(image)

    # --------------------------------------------------------
    # Resize for LIME
    # --------------------------------------------------------

    image_pil = image.resize((224, 224))

    image_np = np.array(image_pil)

    # Convert to float [0, 1]
    image_float = image_np.astype(np.float32) / 255.0

    # --------------------------------------------------------
    # Create LIME explainer
    # --------------------------------------------------------

    explainer = lime_image.LimeImageExplainer(
        verbose=False
    )

    # --------------------------------------------------------
    # Run LIME
    # --------------------------------------------------------

    explanation = explainer.explain_instance(
        image_float,
        predict_batch,
        top_labels=5,
        hide_color=0,
        num_samples=num_samples
    )

    # --------------------------------------------------------
    # Get predicted class
    # --------------------------------------------------------

    predicted_class = explanation.top_labels[0]

    # --------------------------------------------------------
    # Get image + mask
    # --------------------------------------------------------

    temp, mask = explanation.get_image_and_mask(
        predicted_class,
        positive_only=False,
        num_features=15,
        hide_rest=False
    )

    # --------------------------------------------------------
    # Create LIME visualization
    # --------------------------------------------------------

    lime_image_array = mark_boundaries(
        temp,
        mask
    )

    # Convert to uint8
    lime_image_array = (
        lime_image_array * 255
    ).astype(np.uint8)

    explanation_image = Image.fromarray(
        lime_image_array
    )

    # --------------------------------------------------------
    # Get confidence
    # --------------------------------------------------------

    probabilities = predict_batch(
        [image_float]
    )[0]

    confidence = float(
        probabilities[predicted_class]
    )

    return (
        explanation_image,
        predicted_class,
        confidence
    )


# ============================================================
# SAVE LIME IMAGE
# ============================================================

def save_lime_explanation(
    image,
    output_path,
    num_samples=500
):
    """
    Generate and save LIME explanation.
    """

    explanation_image, predicted_class, confidence = explain_image(
        image,
        num_samples=num_samples
    )

    # Make sure output directory exists
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    explanation_image.save(
        output_path
    )

    return {
        "predicted_class": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": round(
            confidence * 100,
            2
        ),
        "output_path": output_path
    }