// ==========================================
// Explainable Diabetic Retinopathy AI
// Frontend JavaScript
// ==========================================

const API_BASE_URL = window.location.origin;


// ==========================================
// DOM ELEMENTS
// ==========================================

const imageInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const fileName = document.getElementById("fileName");

const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");

const analyzeBtn = document.getElementById("analyzeBtn");
const loading = document.getElementById("loading");

const results = document.getElementById("results");

const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");

const probabilities = document.getElementById("probabilities");

const originalImage = document.getElementById("originalImage");
const limeImage = document.getElementById("limeImage");

const errorBox = document.getElementById("errorBox");
const errorMessage = document.getElementById("errorMessage");

const newAnalysisBtn = document.getElementById("newAnalysisBtn");

let selectedFile = null;


// ==========================================
// FILE SELECTION
// ==========================================

imageInput.addEventListener("change", function () {

    if (this.files && this.files.length > 0) {

        handleFile(this.files[0]);

    }

});


// ==========================================
// HANDLE FILE
// ==========================================

function handleFile(file) {

    const allowedTypes = [
        "image/jpeg",
        "image/jpg",
        "image/png"
    ];

    if (!allowedTypes.includes(file.type)) {

        showError(
            "Please upload a JPG, JPEG, or PNG retinal image."
        );

        return;
    }

    const maxSize = 10 * 1024 * 1024;

    if (file.size > maxSize) {

        showError(
            "Image size must be less than 10 MB."
        );

        return;
    }

    selectedFile = file;

    hideError();

    fileName.textContent = file.name;


    const reader = new FileReader();


    reader.onload = function (event) {

        previewImage.src =
            event.target.result;

        previewContainer.classList.remove(
            "hidden"
        );

    };


    reader.readAsDataURL(file);


    analyzeBtn.disabled = false;
}


// ==========================================
// DRAG & DROP
// ==========================================

dropZone.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        dropZone.classList.add(
            "dragover"
        );

    }
);


dropZone.addEventListener(
    "dragleave",
    function () {

        dropZone.classList.remove(
            "dragover"
        );

    }
);


dropZone.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        dropZone.classList.remove(
            "dragover"
        );

        const files =
            event.dataTransfer.files;

        if (files.length > 0) {

            handleFile(files[0]);

        }

    }
);


// ==========================================
// ANALYZE
// ==========================================

analyzeBtn.addEventListener(
    "click",
    analyzeImage
);


async function analyzeImage() {

    if (!selectedFile) {

        showError(
            "Please select a retinal image first."
        );

        return;
    }

    hideError();

    loading.classList.remove(
        "hidden"
    );

    analyzeBtn.disabled = true;

    results.classList.add(
        "hidden"
    );


    try {

        const formData =
            new FormData();


        formData.append(
            "file",
            selectedFile
        );


        const response =
            await fetch(
                `${API_BASE_URL}/predict`,
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            let message =
                "Backend returned an error.";


            try {

                const errorData =
                    await response.json();


                if (errorData.detail) {

                    message =
                        errorData.detail;

                }

            }

            catch (e) {}


            throw new Error(
                message
            );
        }


        const data =
            await response.json();


        console.log(
            "FastAPI Response:",
            data
        );


        displayResults(data);

    }

    catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        showError(
            "Unable to connect to the AI backend. " +
            error.message
        );

    }

    finally {

        loading.classList.add(
            "hidden"
        );

        analyzeBtn.disabled = false;

    }
}


// ==========================================
// DISPLAY RESULTS
// ==========================================

function displayResults(data) {

    const predictionData =
        data.prediction;


    // ======================================
    // PREDICTION
    // ======================================

    if (predictionData) {

        // ----------------------------------
        // DR GRADE
        // ----------------------------------

        const grade =
            Number(
                predictionData.grade
            );


        if (
            !isNaN(grade) &&
            grade >= 0 &&
            grade <= 4
        ) {

            // Main prediction text
            prediction.textContent =
                `Grade ${grade} — ` +
                `${predictionData.class_name}`;

        }

        else {

            prediction.textContent =
                predictionData.class_name ||
                "Unable to determine grade";

        }


        // ----------------------------------
        // CONFIDENCE
        // ----------------------------------

        const confidenceValue =
            Number(
                predictionData.confidence
            );


        if (!isNaN(confidenceValue)) {

            confidence.textContent =
                `${confidenceValue.toFixed(2)}%`;

        }

        else {

            confidence.textContent =
                "--%";

        }


        // ----------------------------------
        // PROBABILITIES
        // ----------------------------------

        displayProbabilities(
            predictionData.probabilities
        );

    }

    else {

        prediction.textContent =
            "Unable to determine grade";

        confidence.textContent =
            "--%";

        probabilities.innerHTML =
            "<p>Probability data unavailable.</p>";

    }


    // ======================================
    // ORIGINAL IMAGE
    // ======================================

    const reader =
        new FileReader();


    reader.onload =
        function (event) {

            originalImage.src =
                event.target.result;

        };


    reader.readAsDataURL(
        selectedFile
    );


    // ======================================
    // LIME IMAGE
    // ======================================

    if (
        data.lime &&
        data.lime.available &&
        data.lime.image_data
    ) {

        // Direct Base64 image
        limeImage.src =
            data.lime.image_data;


        limeImage.alt =
            "LIME explanation for " +
            `Grade ${data.lime.grade}`;


        console.log(
            "✅ LIME image displayed."
        );


        // ----------------------------------
        // LIME INFO
        // ----------------------------------

        console.log(
            "LIME Grade:",
            data.lime.grade
        );

        console.log(
            "LIME Class:",
            data.lime.class_name
        );

    }

    else {

        limeImage.removeAttribute(
            "src"
        );


        limeImage.alt =
            "LIME explanation unavailable";


        console.warn(
            "⚠️ LIME explanation unavailable."
        );

    }


    // ======================================
    // SHOW RESULTS
    // ======================================

    results.classList.remove(
        "hidden"
    );


    setTimeout(
        function () {

            results.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        },
        100
    );
}


// ==========================================
// PROBABILITIES
// ==========================================

function displayProbabilities(
    probabilityData
) {

    probabilities.innerHTML =
        "";


    if (
        !probabilityData ||
        typeof probabilityData !== "object"
    ) {

        probabilities.innerHTML =
            "<p>Probability data unavailable.</p>";

        return;
    }


    Object.entries(
        probabilityData
    ).forEach(
        ([className, rawValue]) => {

            let value =
                Number(rawValue);


            if (isNaN(value)) {

                value = 0;

            }


            value =
                Math.max(
                    0,
                    Math.min(
                        100,
                        value
                    )
                );


            // --------------------------------
            // ROW
            // --------------------------------

            const row =
                document.createElement(
                    "div"
                );


            row.className =
                "probability-row";


            // --------------------------------
            // LABEL
            // --------------------------------

            const label =
                document.createElement(
                    "div"
                );


            label.className =
                "probability-label";


            label.textContent =
                className;


            // --------------------------------
            // PROGRESS
            // --------------------------------

            const progress =
                document.createElement(
                    "div"
                );


            progress.className =
                "progress";


            // --------------------------------
            // BAR
            // --------------------------------

            const bar =
                document.createElement(
                    "div"
                );


            bar.className =
                "progress-bar";


            bar.style.width =
                `${value}%`;


            progress.appendChild(
                bar
            );


            // --------------------------------
            // VALUE
            // --------------------------------

            const valueText =
                document.createElement(
                    "div"
                );


            valueText.className =
                "probability-value";


            valueText.textContent =
                `${value.toFixed(2)}%`;


            // --------------------------------
            // ADD ELEMENTS
            // --------------------------------

            row.appendChild(
                label
            );


            row.appendChild(
                progress
            );


            row.appendChild(
                valueText
            );


            probabilities.appendChild(
                row
            );

        }
    );
}


// ==========================================
// ERROR
// ==========================================

function showError(message) {

    errorMessage.textContent =
        message;


    errorBox.classList.remove(
        "hidden"
    );


    errorBox.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });

}


function hideError() {

    errorBox.classList.add(
        "hidden"
    );


    errorMessage.textContent =
        "";

}


// ==========================================
// NEW ANALYSIS
// ==========================================

newAnalysisBtn.addEventListener(
    "click",
    function () {

        selectedFile = null;


        imageInput.value =
            "";


        fileName.textContent =
            "No image selected";


        previewImage.removeAttribute(
            "src"
        );


        previewContainer.classList.add(
            "hidden"
        );


        results.classList.add(
            "hidden"
        );


        limeImage.removeAttribute(
            "src"
        );


        originalImage.removeAttribute(
            "src"
        );


        probabilities.innerHTML =
            "";


        analyzeBtn.disabled =
            true;


        hideError();


        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });

    }
);


// ==========================================
// BACKEND CHECK
// ==========================================

async function checkBackend() {

    try {

        const response =
            await fetch(
                `${API_BASE_URL}/health`
            );


        if (response.ok) {

            console.log(
                "✅ FastAPI backend connected."
            );

        }

        else {

            console.warn(
                "⚠️ FastAPI backend responded with an error."
            );

        }

    }

    catch (error) {

        console.warn(
            "⚠️ FastAPI backend is not reachable.",
            error
        );

    }

}


checkBackend();