import os
import gc
import numpy as np
from PIL import Image

from lime import lime_image
from skimage.segmentation import mark_boundaries

from backend.model import predict_batch, CLASS_NAMES


# ============================================================
# LIME EXPLAINER
# ============================================================

def explain_image(image, num_samples=25):
    """
    Generate a memory-optimized LIME explanation
    for a diabetic retinopathy retinal image.

    Optimized for low-memory cloud deployment.

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

    # --------------------------------------------------------
    # Resize image for LIME
    # --------------------------------------------------------

    image_pil = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    image_np = np.asarray(
        image_pil,
        dtype=np.float32
    )

    # Convert pixel values to [0, 1]
    image_float = image_np / 255.0

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
        num_samples=num_samples,
        batch_size=5
    )

    # --------------------------------------------------------
    # Get predicted class
    # --------------------------------------------------------

    predicted_class = explanation.top_labels[0]

    # --------------------------------------------------------
    # Get image and mask
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

    # Convert to uint8 safely
    lime_image_array = (
        lime_image_array * 255.0
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

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

    # --------------------------------------------------------
    # Free memory
    # --------------------------------------------------------

    del image_pil
    del image_np
    del image_float
    del explanation
    del temp
    del mask
    del lime_image_array
    del probabilities

    gc.collect()

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
    num_samples=25
):
    """
    Generate and save LIME explanation.
    """

    explanation_image, predicted_class, confidence = explain_image(
        image,
        num_samples=num_samples
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Save optimized JPEG
    # --------------------------------------------------------

    explanation_image.save(
        output_path,
        format="JPEG",
        quality=85,
        optimize=True
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "predicted_class": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": round(
            confidence * 100,
            2
        ),
        "output_path": output_path
    }
