import os

class Config:
    """Configuration settings for the medical chatbot application."""
    
    # General settings
    DEBUG = os.getenv("DEBUG", "True") == "True"
    SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
    
    # Model settings
    SYMPTOM_CLASSIFIER_MODEL_PATH = os.getenv("SYMPTOM_CLASSIFIER_MODEL_PATH", "models/symptom_classifier.h5")
    DRUG_RECOMMENDER_MODEL_PATH = os.getenv("DRUG_RECOMMENDER_MODEL_PATH", "models/drug_recommender.h5")
    
    # API settings
    API_VERSION = "v1"
    API_PREFIX = os.getenv("API_PREFIX", "/api")
    
    # Database settings (if applicable)
    DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///data/database.db")
    
    # Other settings
    MAX_INPUT_LENGTH = 100
    MAX_RESPONSE_LENGTH = 200
    LANGUAGE = "en"  # Default language for responses