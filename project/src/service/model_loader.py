import joblib

from src.service.config import settings


artifacts = joblib.load(settings.MODEL_PATH)

model = artifacts["model"]
feature_columns = artifacts["feature_columns"]