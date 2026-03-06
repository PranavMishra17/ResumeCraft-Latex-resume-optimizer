import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    """Configuration for LLM providers"""
    
    # Active Provider (can be azure, openai, gemini, anthropic)
    ACTIVE_PROVIDER = os.getenv("LLM_PROVIDER", "azure").lower()
    
    # Common Settings
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))

    # Azure OpenAI Configuration
    AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    # Google Gemini Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    
    # Anthropic Default Configuration
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

    @classmethod
    def validate(cls):
        if cls.ACTIVE_PROVIDER == "azure" and not cls.AZURE_API_KEY:
            logger.error("AZURE_API_KEY must be set in environment variables when provider is azure.")
            sys.exit(1)
        elif cls.ACTIVE_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY must be set when provider is openai.")
            sys.exit(1)
        elif cls.ACTIVE_PROVIDER == "gemini" and not cls.GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY must be set when provider is gemini.")
            sys.exit(1)
        elif cls.ACTIVE_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            logger.error("ANTHROPIC_API_KEY must be set when provider is anthropic.")
            sys.exit(1)