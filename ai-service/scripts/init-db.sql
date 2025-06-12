# ai-service/scripts/init-db.sql
-- Initialize database with required extensions and basic setup

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- Create AI service user if not exists
DO $
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'aiuser') THEN
        CREATE ROLE aiuser LOGIN PASSWORD 'aipass';
    END IF;
END
$;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE taskmaster_ai TO aiuser;
GRANT ALL PRIVILEGES ON SCHEMA public TO aiuser;

-- Create indexes for better performance
-- These will be created by Alembic migrations, but having them here as backup

-- Indexes for user_behavior_patterns
CREATE INDEX IF NOT EXISTS idx_user_behavior_patterns_user_id 
ON user_behavior_patterns(user_id);

CREATE INDEX IF NOT EXISTS idx_user_behavior_patterns_pattern_type 
ON user_behavior_patterns(pattern_type);

CREATE INDEX IF NOT EXISTS idx_user_behavior_patterns_created_at 
ON user_behavior_patterns(created_at);

-- Indexes for task_embeddings
CREATE INDEX IF NOT EXISTS idx_task_embeddings_user_id 
ON task_embeddings(user_id);

CREATE INDEX IF NOT EXISTS idx_task_embeddings_category 
ON task_embeddings(category);

-- Vector similarity indexes (will be created after table creation)
-- CREATE INDEX idx_task_embeddings_embedding ON task_embeddings 
-- USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);