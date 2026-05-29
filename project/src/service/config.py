from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "AI credit risk assessment"
    MODEL_PATH: str = "./artifacts/BestModel/best_model.joblib"

    class Config:
        env_file = ".env"


settings = Settings()