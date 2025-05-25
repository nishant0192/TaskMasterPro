# ai-service/app/services/rag_service.py
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.client import Client as ChromaClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sqlmodel import Session, select

from app.core.config import settings
from app.db.base import engine
from app.db.models import TrainingData

class RAGService:
    def __init__(self):
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        self.generator = pipeline(
            "text2text-generation",
            model=settings.GENERATOR_MODEL_NAME,
            device=0
        )
        self.client = chromadb.HttpClient(
            host=settings.CHROMA_DB_HOST,
            port=settings.CHROMA_DB_PORT
        )
        try:
            self.col = self.client.get_collection(name="tasks")
        except Exception:
            self.col = self.client.create_collection(
                name="tasks",
                embedding_function=self._emb_fn
            )

    def _emb_fn(self, texts):
        return self.embedder.encode(texts, convert_to_numpy=True).tolist()

    def ingest_all(self):
        with Session(engine) as session:
            rows = session.exec(
                select(TrainingData).where(TrainingData.task_description != None)
            ).all()
        ids, docs, metas = [], [], []
        for r in rows:
            if not r.task_description: continue
            ids.append(r.id)
            docs.append(r.task_description)
            metas.append({"task_id": r.task_id, "user_id": r.user_id})
        if ids:
            self.col.add(ids=ids, documents=docs, metadatas=metas)

    def retrieve(self, query: str, top_k: int = 5):
        res = self.col.query(query_texts=[query], n_results=top_k)
        out = []
        for i, doc in enumerate(res["documents"][0]):
            out.append({
                "id": res["ids"][0][i],
                "text": doc,
                **res["metadatas"][0][i]
            })
        return out

    def generate(self, query: str, contexts):
        ctx = "\n".join(f"- {c}" for c in contexts)
        prompt = f"Context:\n{ctx}\n\nTask: {query}\n\nSuggestion:"
        gen = self.generator(prompt, max_length=256, do_sample=False)
        return gen[0]["generated_text"]

    def rag(self, query: str, top_k: int = 5):
        docs = self.retrieve(query, top_k)
        texts = [d["text"] for d in docs]
        suggestion = self.generate(query, texts)
        return {"retrieved": docs, "suggestion": suggestion}
