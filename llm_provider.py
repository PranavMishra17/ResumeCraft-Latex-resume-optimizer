import sys
import logging
from config import Config

logger = logging.getLogger(__name__)

def get_llm_client():
    """Initialize and return LLM client based on configuration"""
    
    Config.validate()
    provider = Config.ACTIVE_PROVIDER
    
    if provider == "azure":
        try:
            from langchain_openai import AzureChatOpenAI
            client = AzureChatOpenAI(
                azure_deployment=Config.AZURE_DEPLOYMENT,
                api_key=Config.AZURE_API_KEY,
                api_version=Config.AZURE_API_VERSION,
                azure_endpoint=Config.AZURE_ENDPOINT,
                temperature=Config.TEMPERATURE
            )
            logger.info(f"Azure OpenAI client ({Config.AZURE_DEPLOYMENT}) initialized successfully!")
            return client
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI client: {e}")
            sys.exit(1)
            
    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            client = ChatOpenAI(
                model=Config.OPENAI_MODEL,
                api_key=Config.OPENAI_API_KEY,
                temperature=Config.TEMPERATURE
            )
            logger.info(f"OpenAI client ({Config.OPENAI_MODEL}) initialized successfully!")
            return client
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            sys.exit(1)
            
    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            client = ChatGoogleGenerativeAI(
                model=Config.GEMINI_MODEL,
                google_api_key=Config.GEMINI_API_KEY,
                temperature=Config.TEMPERATURE,
                convert_system_message_to_human=True # Required by some langchain-google-genai versions
            )
            logger.info(f"Google Gemini client ({Config.GEMINI_MODEL}) initialized successfully!")
            return client
        except Exception as e:
            logger.error(f"Error initializing Google Gemini client: {e}")
            logger.error("You may need to install the package: pip install langchain-google-genai")
            sys.exit(1)
            
    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            client = ChatAnthropic(
                model_name=Config.ANTHROPIC_MODEL,
                anthropic_api_key=Config.ANTHROPIC_API_KEY,
                temperature=Config.TEMPERATURE
            )
            logger.info(f"Anthropic Claude client ({Config.ANTHROPIC_MODEL}) initialized successfully!")
            return client
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {e}")
            logger.error("You may need to install the package: pip install langchain-anthropic")
            sys.exit(1)
            
    else:
        logger.error(f"Unknown provider '{provider}'. Please set LLM_PROVIDER to azure, openai, gemini, or anthropic.")
        sys.exit(1)
