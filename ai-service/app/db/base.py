# ai-service/app/db/base.py
from sqlmodel import SQLModel, create_engine, Session

from app.core.config import settings

engine = create_engine(
    settings.AI_DATABASE_URL,
    echo=False,
    connect_args={},  # e.g., {"check_same_thread": False} for SQLite
)

def get_session():
    with Session(engine) as session:
        yield session
