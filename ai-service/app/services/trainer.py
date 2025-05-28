from app.celery_app import celery
from app.models.job import Job, JobStatus, init_models
from app.db.session import AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import asyncio

@celery.task(bind=True)
def train_model(self, training_data: dict):
    job_id = self.request.id
    # update job status in DB
    asyncio.run(_update_job(job_id, status=JobStatus.RUNNING))
    try:
        # perform training (placeholder)
        result = {"metrics": {"loss": 0.1}}
        asyncio.run(_update_job(job_id, status=JobStatus.SUCCESS, result=result))
        return result
    except Exception as e:
        asyncio.run(_update_job(job_id, status=JobStatus.FAILED, error={"msg": str(e)}))
        raise

async def _update_job(job_id: str, **fields):
    async with AsyncSessionLocal() as session:
        stmt = update(Job).where(Job.id == job_id).values(**fields)
        await session.execute(stmt)
        await session.commit()
