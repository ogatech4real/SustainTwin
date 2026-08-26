import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import uvicorn


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "SUSTAINTWIN_MODEL_PATH",
        BASE_DIR / "fan_fault_classifier.joblib",
    )
)

API_HOST = os.getenv("SUSTAINTWIN_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("SUSTAINTWIN_API_PORT", "8000"))

MODEL_NAME = "SustainTwin Random Forest Fault Classifier"
API_VERSION = "3.0.0"


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

DEFAULT_FEATURE_COLUMNS = [
    "temp",
    "rpm",
    "current",
    "vibration",
    "power",
    "energy",
    "coolingRate",
    "thermalEff",
    "co2",
]


# ============================================================
# PYDANTIC REQUEST AND RESPONSE MODELS
# ============================================================

class ScoreRequest(BaseModel):
    """
    Process-state feature vector received from Node-RED.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "temp": 38.5,
                "rpm": 1950,
                "current": 1.027,
                "vibration": 0.0254,
                "power": 207.97,
                "energy": 14.12,
                "coolingRate": 0.0135,
                "thermalEff": 6.49,
                "co2": 3.29,
            }
        },
    )

    temp: float = Field(
        ...,
        description="Process temperature in degrees Celsius",
    )

    rpm: float = Field(
        ...,
        ge=0,
        description="Fan rotational speed in revolutions per minute",
    )

    current: float = Field(
        ...,
        ge=0,
        description="Motor current in amperes",
    )

    vibration: float = Field(
        ...,
        ge=0,
        description="Vibration magnitude in g RMS",
    )

    power: float = Field(
        ...,
        ge=0,
        description="Instantaneous electrical power in watts",
    )

    energy: float = Field(
        ...,
        ge=0,
        description="Cumulative energy consumption in watt-hours",
    )

    coolingRate: float = Field(
        ...,
        description=(
            "Cooling rate in degrees Celsius per second. "
            "Positive values indicate falling temperature."
        ),
    )

    thermalEff: float = Field(
        ...,
        ge=0,
        description=(
            "Legacy thermal-efficiency index used by the trained model"
        ),
    )

    co2: float = Field(
        ...,
        ge=0,
        description="Estimated cumulative carbon emissions in grams",
    )


class ScoreResponse(BaseModel):
    fault_code: int
    fault_label: str
    confidence: float
    confidence_percent: float
    severity: str
    advice: str
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    model_name: str
    api_version: str
    feature_columns: list[str]


# ============================================================
# GLOBAL MODEL STATE
# ============================================================

model_pipeline: Any | None = None
feature_columns: list[str] = DEFAULT_FEATURE_COLUMNS.copy()
model_metadata: dict[str, Any] = {}


# ============================================================
# MODEL LOADING
# ============================================================

def load_model() -> None:
    """
    Load the trained model and metadata from disk.

    Expected joblib structure:

    {
        "pipeline": fitted_sklearn_pipeline,
        "feature_cols": [...]
    }
    """

    global model_pipeline
    global feature_columns
    global model_metadata

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    if isinstance(artifact, dict):
        if "pipeline" not in artifact:
            raise ValueError(
                "Model artifact does not contain a 'pipeline' entry."
            )

        model_pipeline = artifact["pipeline"]

        saved_features = artifact.get(
            "feature_cols",
            DEFAULT_FEATURE_COLUMNS,
        )

        feature_columns = list(saved_features)

        model_metadata = {
            key: value
            for key, value in artifact.items()
            if key not in {"pipeline", "feature_cols"}
        }

    else:
        # Fallback support if only the fitted estimator/pipeline was saved.
        model_pipeline = artifact
        feature_columns = DEFAULT_FEATURE_COLUMNS.copy()
        model_metadata = {}

    if not hasattr(model_pipeline, "predict"):
        raise TypeError(
            "Loaded model does not provide a predict() method."
        )

    print(f"[MODEL] Loaded model from: {MODEL_PATH}")
    print(f"[MODEL] Features: {feature_columns}")


# ============================================================
# FAULT INTERPRETATION
# ============================================================

def get_fault_label(fault_code: int) -> str:
    """
    Map the current binary model output to a readable label.

    The current training target is binary:
    0 = Normal
    1 = Fault detected
    """

    labels = {
        0: "Normal",
        1: "Fault Detected",
    }

    return labels.get(
        int(fault_code),
        f"Fault Class {fault_code}",
    )


def get_severity(
    fault_code: int,
    confidence: float,
) -> str:
    if fault_code == 0:
        return "None"

    if confidence >= 0.90:
        return "High"

    if confidence >= 0.70:
        return "Medium"

    return "Low"


def get_maintenance_advice(
    fault_code: int,
    confidence: float,
    request: ScoreRequest,
) -> str:
    if fault_code == 0:
        return (
            "No immediate maintenance action is required. "
            "Continue normal monitoring."
        )

    indicators: list[str] = []

    if request.vibration >= 0.06:
        indicators.append(
            "elevated vibration"
        )

    if request.current >= 1.50:
        indicators.append(
            "high motor current"
        )

    if request.temp >= 45.0:
        indicators.append(
            "high process temperature"
        )

    if request.coolingRate < 0:
        indicators.append(
            "rising process temperature"
        )

    if indicators:
        indicator_text = ", ".join(indicators)

        return (
            f"Fault conditions are associated with {indicator_text}. "
            "Inspect the fan, mechanical loading, motor current, "
            "cooling path, and thermal operating conditions."
        )

    if confidence >= 0.90:
        return (
            "A high-confidence abnormal condition has been detected. "
            "Inspect the cooling system, motor loading, vibration level, "
            "and electrical connections before continued operation."
        )

    return (
        "An abnormal operating pattern has been detected. "
        "Continue monitoring and schedule an inspection if the "
        "condition persists."
    )


# ============================================================
# MODEL INFERENCE
# ============================================================

def request_to_dataframe(
    request: ScoreRequest,
) -> pd.DataFrame:
    request_values = request.model_dump()

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in request_values
    ]

    if missing_features:
        raise ValueError(
            "Request is missing model features: "
            + ", ".join(missing_features)
        )

    row = {
        feature: float(request_values[feature])
        for feature in feature_columns
    }

    dataframe = pd.DataFrame(
        [row],
        columns=feature_columns,
    )

    values = dataframe.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Request contains NaN or infinite feature values."
        )

    return dataframe


def predict_fault(
    request: ScoreRequest,
) -> tuple[int, float]:
    if model_pipeline is None:
        raise RuntimeError(
            "The machine-learning model is not loaded."
        )

    dataframe = request_to_dataframe(request)

    prediction = model_pipeline.predict(dataframe)

    if len(prediction) != 1:
        raise RuntimeError(
            "Unexpected prediction output length."
        )

    fault_code = int(prediction[0])
    confidence = 1.0

    if hasattr(model_pipeline, "predict_proba"):
        probabilities = model_pipeline.predict_proba(dataframe)

        if probabilities.shape[0] == 1:
            estimator_classes = getattr(
                model_pipeline,
                "classes_",
                None,
            )

            if estimator_classes is not None:
                classes = list(estimator_classes)

                if fault_code in classes:
                    class_index = classes.index(fault_code)
                    confidence = float(
                        probabilities[0][class_index]
                    )
                else:
                    confidence = float(
                        probabilities[0].max()
                    )
            else:
                confidence = float(
                    probabilities[0].max()
                )

    confidence = max(
        0.0,
        min(1.0, confidence),
    )

    return fault_code, confidence


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SustainTwin AI Fault Intelligence API",
    description=(
        "Machine-learning inference service for the "
        "SustainTwin digital twin framework."
    ),
    version=API_VERSION,
)


@app.on_event("startup")
def startup_event() -> None:
    try:
        load_model()
    except Exception as error:
        print(f"[MODEL] Loading failed: {error}")
        raise


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "SustainTwin AI Fault Intelligence API",
        "status": "running",
        "api_version": API_VERSION,
        "model_loaded": model_pipeline is not None,
        "documentation": "/docs",
        "health_endpoint": "/health",
        "scoring_endpoint": "/score",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(
        status=(
            "healthy"
            if model_pipeline is not None
            else "model unavailable"
        ),
        model_loaded=model_pipeline is not None,
        model_path=str(MODEL_PATH),
        model_name=MODEL_NAME,
        api_version=API_VERSION,
        feature_columns=feature_columns,
    )


@app.post(
    "/score",
    response_model=ScoreResponse,
)
def score(request: ScoreRequest) -> ScoreResponse:
    try:
        fault_code, confidence = predict_fault(request)

        fault_label = get_fault_label(fault_code)

        severity = get_severity(
            fault_code,
            confidence,
        )

        advice = get_maintenance_advice(
            fault_code,
            confidence,
            request,
        )

        return ScoreResponse(
            fault_code=fault_code,
            fault_label=fault_label,
            confidence=round(confidence, 6),
            confidence_percent=round(
                confidence * 100,
                2,
            ),
            severity=severity,
            advice=advice,
            model_name=MODEL_NAME,
            model_version=str(
                model_metadata.get(
                    "model_version",
                    "binary-random-forest-v1",
                )
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {error}",
        ) from error


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "serve_model:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )