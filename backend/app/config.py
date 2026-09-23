import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "PPG Glucose Prediction System"
    API_V1_STR: str = "/api"
    
    # Database
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "password123"
    DB_HOST: str = "localhost"
    DB_PORT: str = "5432"
    DB_NAME: str = "ppg_glucose_db"
    DATABASE_URL: str = ""
    
    # ML Models Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH: str = os.path.join(BASE_DIR, "ml_models", "xgboost_glucose_model.pkl")
    SCALER_PATH: str = os.path.join(BASE_DIR, "ml_models", "scaler_params.json")
    
    # PPG Constants
    FS: float = 400.0  # Sampling rate 400Hz
    WINDOW_DURATION_SEC: float = 15.0  # 15 second windows
    
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    def get_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

settings = Settings()
