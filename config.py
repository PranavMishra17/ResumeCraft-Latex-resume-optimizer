import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    """Configuration for LLM providers"""
    
    # Azure OpenAI Configuration (Default)
    AZURE_DEPLOYMENT = "VARELab-GPT4o"
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "enter_fallback_here") # Either setup the environment variable or provide a fallback
    AZURE_API_VERSION = "2024-08-01-preview"
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "enter_fallback_here") # Either setup the environment variable or provide a fallback
    TEMPERATURE = 0.2
    
    """OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = "gpt-4o"
    TEMPERATURE = 0.2
    """
    
    """Claude Configuration  
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL = "claude-3-sonnet-20240229"
    TEMPERATURE = 0.2
    """

def get_llm_client():
    """Initialize and return LLM client based on configuration"""
    
    # Azure OpenAI (Default)
    if Config.AZURE_API_KEY and Config.AZURE_ENDPOINT:
        try:
            from langchain_openai import AzureChatOpenAI
            client = AzureChatOpenAI(
                azure_deployment=Config.AZURE_DEPLOYMENT,
                api_key=Config.AZURE_API_KEY,
                api_version=Config.AZURE_API_VERSION,
                azure_endpoint=Config.AZURE_ENDPOINT,
                temperature=Config.TEMPERATURE
            )
            logger.info("Azure OpenAI client initialized successfully!")
            return client
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI client: {e}")
            sys.exit(1)
    
    """OpenAI Alternative
    elif Config.OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            client = ChatOpenAI(
                api_key=Config.OPENAI_API_KEY,
                model=Config.OPENAI_MODEL,
                temperature=Config.TEMPERATURE
            )
            logger.info("OpenAI client initialized successfully!")
            return client
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            sys.exit(1)
    """
    
    """Claude Alternative
    elif Config.ANTHROPIC_API_KEY:
        try:
            from langchain_anthropic import ChatAnthropic
            client = ChatAnthropic(
                api_key=Config.ANTHROPIC_API_KEY,
                model=Config.CLAUDE_MODEL,
                temperature=Config.TEMPERATURE
            )
            logger.info("Claude client initialized successfully!")
            return client
        except Exception as e:
            logger.error(f"Error initializing Claude client: {e}")
            sys.exit(1)
    """
    
    logger.error("No valid API keys found. Please set one of:")
    logger.error("- AZURE_OPENAI_API_KEY and AZURE_ENDPOINT")
    logger.error("- OPENAI_API_KEY") 
    logger.error("- ANTHROPIC_API_KEY")
    sys.exit(1)