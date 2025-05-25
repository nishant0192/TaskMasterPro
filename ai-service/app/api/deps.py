# ai-service/app/api/deps.py
from sqlmodel import Session
from fastapi import Depends

from app.db.base import get_session

def get_db_session() -> Session:
    return Depends(get_session)
