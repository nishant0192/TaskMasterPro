# File: api/endpoints/generate_subatsks.py

import os
# Set HF_TRUST_REMOTE_CODE to trust custom code from remote repositories (required for wiki_dpr)
os.environ["HF_TRUST_REMOTE_CODE"] = "1"

import uuid
import datetime
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

# SQLAlchemy imports for PostgreSQL integration
from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

# Transformer imports for ML model integration
import torch
from transformers import (
    RagTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# -------------------------------------------------
# Logging Configuration
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Database Setup (SQLAlchemy)
# -------------------------------------------------
DATABASE_URL = "postgresql://postgres:nishant%3F%40980@localhost:5432/taskmasterpro"

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Minimal models (adapt as needed to your full Prisma schema)
class Task(Base):
    __tablename__ = "Task"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    description = Column(String)
    creator_id = Column(String)  # In production, use a ForeignKey to your User table
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    subtasks = relationship("Subtask", back_populates="task", cascade="all, delete-orphan")


class Subtask(Base):
    __tablename__ = "Subtask"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    is_completed = Column(Boolean, default=False)
    order = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    task_id = Column(String, ForeignKey("Task.id", ondelete="CASCADE"), nullable=False)
    task = relationship("Task", back_populates="subtasks")


# In production, manage schema migrations externally rather than calling create_all
Base.metadata.create_all(bind=engine)

# Dependency to get a database session per request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------------------------
# ML Model: Personalized Subtask Generator with Optional RAG
# -------------------------------------------------
class PersonalizedSubtaskGenerator:
    def __init__(self, use_rag: bool = True):
        self.use_rag = use_rag
        try:
            if self.use_rag:
                model_name = "facebook/rag-token-nq"
                self.tokenizer = RagTokenizer.from_pretrained(model_name)
                # Pass trust_remote_code via the environment variable we set above
                self.retriever = RagRetriever.from_pretrained(
                    model_name,
                    index_name="exact",
                    use_dummy_dataset=True,
                )
                self.model = RagSequenceForGeneration.from_pretrained(
                    model_name,
                )
            else:
                fallback_model_name = "t5-base"
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model_name)
        except Exception as e:
            logger.error("Error loading ML model: %s", e)
            raise

    def generate_personalized_subtasks(
        self, task_title: str, task_description: str, user_context: str, num_subtasks: int = 3
    ) -> List[str]:
        prompt = (
            f"Task: {task_title}\n"
            f"Description: {task_description}\n"
            f"User Context: {user_context}\n"
            f"Generate {num_subtasks} actionable subtasks for this task."
        )
        try:
            if self.use_rag:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    num_beams=num_subtasks,
                    num_return_sequences=num_subtasks,
                    max_length=64,
                    early_stopping=True,
                )
                subtasks = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            else:
                inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                generated_ids = self.model.generate(
                    inputs,
                    max_length=64,
                    num_beams=num_subtasks,
                    num_return_sequences=num_subtasks,
                    early_stopping=True,
                )
                subtasks = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return subtasks
        except Exception as e:
            logger.error("Error during subtask generation: %s", e)
            raise HTTPException(status_code=500, detail="Error generating subtasks")


# Global generator instance (set use_rag=True or False as needed)
generator = PersonalizedSubtaskGenerator(use_rag=True)

# -------------------------------------------------
# Pydantic Models for Request and Response
# -------------------------------------------------
class TaskGenerationRequest(BaseModel):
    user_id: str
    task_id: str   # Parent task ID to attach generated subtasks
    title: str
    description: str
    number_of_subtasks: Optional[int] = 3


class SubtaskResponse(BaseModel):
    id: str
    title: str
    order: int


class TaskGenerationResponse(BaseModel):
    subtasks: List[SubtaskResponse]

# -------------------------------------------------
# Helper: Get User Context from DB
# -------------------------------------------------
def get_user_context(db: Session, user_id: str) -> str:
    """
    Retrieves user context by aggregating titles of recent tasks created by the user.
    """
    try:
        tasks = (
            db.query(Task)
            .filter(Task.creator_id == user_id)
            .order_by(Task.created_at.desc())
            .limit(5)
            .all()
        )
        context = " | ".join([task.title for task in tasks]) if tasks else "No historical tasks"
        return context
    except Exception as e:
        logger.error("Error retrieving user context: %s", e)
        return "No historical tasks"

# -------------------------------------------------
# FastAPI Router Setup
# -------------------------------------------------
router = APIRouter()

@router.post("/generate-personalized-subtasks", response_model=TaskGenerationResponse)
async def generate_personalized_subtasks_endpoint(
    request: TaskGenerationRequest,
    db: Session = Depends(get_db),
):
    try:
        # Retrieve additional context for the user
        user_context = get_user_context(db, request.user_id)

        # Generate personalized subtasks using the ML model
        generated_titles = generator.generate_personalized_subtasks(
            task_title=request.title,
            task_description=request.description,
            user_context=user_context,
            num_subtasks=request.number_of_subtasks,
        )

        # Ensure the parent task exists
        parent_task = db.query(Task).filter(Task.id == request.task_id).first()
        if not parent_task:
            raise HTTPException(status_code=404, detail="Parent task not found")

        # Create Subtask records in the database
        subtask_objects = []
        for order, title in enumerate(generated_titles):
            new_subtask = Subtask(
                title=title,
                order=order,
                task_id=request.task_id,
            )
            db.add(new_subtask)
            subtask_objects.append(new_subtask)
        db.commit()

        # Refresh objects to obtain their IDs
        for subtask in subtask_objects:
            db.refresh(subtask)

        response_subtasks = [
            SubtaskResponse(id=subtask.id, title=subtask.title, order=subtask.order)
            for subtask in subtask_objects
        ]
        return TaskGenerationResponse(subtasks=response_subtasks)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error("Error in generate-personalized-subtasks endpoint: %s", e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")
