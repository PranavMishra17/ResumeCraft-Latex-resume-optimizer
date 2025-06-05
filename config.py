import argparse
import os
import sys
import re
import subprocess
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
import logging
from dataclasses import dataclass
from typing import List, Dict, Set
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration for Azure OpenAI"""
    AZURE_DEPLOYMENT = "VARELab-GPT4o"
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "" ) # Set this in your environment variables

    if not AZURE_API_KEY:
        logger.error("AZURE_API_KEY must be set in environment variables.")
        sys.exit(1)
    AZURE_API_VERSION = "2024-08-01-preview"
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
    TEMPERATURE = 0.2
