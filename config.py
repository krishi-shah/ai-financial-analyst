"""
Configuration — all secrets are loaded from environment variables.
Copy `.env.example` to `.env` and fill in your keys.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
YAHOO_FINANCE_API = os.getenv("YAHOO_FINANCE_API", "")

SEC_BASE_URL = "https://www.sec.gov/Archives/"
