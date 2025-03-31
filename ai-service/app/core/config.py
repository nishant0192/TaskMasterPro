import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("AI_DATABASE_URL")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
