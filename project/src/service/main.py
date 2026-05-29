from fastapi import FastAPI, HTTPException
import pandas as pd

from src.service.schemas import (
    PredictRequest,
    PredictResponse
)

from src.service.model_loader import (
    model,
    feature_columns
)

from src.service.logger import setup_logger

logger = setup_logger()

app = FastAPI(title="AI credit risk assessment")


@app.get("/")
def root():
    return {"message": "Service is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "features": feature_columns,
        "num_features": len(feature_columns)
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    try:

        input_data = request.features

        logger.info(f"Received request: {input_data}")

        missing_features = [
            col for col in feature_columns
            if col not in input_data
        ]

        if missing_features:
            raise HTTPException(
                status_code=400,
                detail={"missing_features": missing_features,
                        "error": "Missing features"}
            )

        df = pd.DataFrame([input_data])

        df = df[feature_columns]

        prediction = model.predict(df)[0]

        probability = model.predict_proba(df)[0].max()

        logger.info(
            f"Prediction={prediction}, Probability={probability}"
        )

        return PredictResponse(
            prediction=int(prediction),
            probability=float(probability)
        )

    except HTTPException:
        raise

    except Exception as e:

        logger.error(str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )