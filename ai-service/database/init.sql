-- ai-service/database/init.sql
-- Initialize AI Service Database

-- Create extensions if not exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create custom types
DO $$ BEGIN
    CREATE TYPE job_status AS ENUM ('pending', 'running', 'completed', 'failed');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Grant permissions (adjust based on your user setup)
GRANT ALL PRIVILEGES ON DATABASE taskmaster_ai TO ai_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ai_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ai_user;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_task_embeddings_user_created 
    ON task_embeddings(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_user_behavior_patterns_composite 
    ON user_behavior_patterns(user_id, pattern_type, confidence_score DESC);

CREATE INDEX IF NOT EXISTS idx_ai_predictions_composite 
    ON ai_predictions(user_id, prediction_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_user_interaction_logs_composite 
    ON user_interaction_logs(user_id, interaction_type, interaction_timestamp DESC);

-- Create materialized view for user productivity summary
CREATE MATERIALIZED VIEW IF NOT EXISTS user_productivity_summary AS
SELECT 
    upm.user_id,
    DATE_TRUNC('week', upm.metric_date) as week,
    AVG(upm.productivity_score) as avg_productivity,
    AVG(upm.tasks_completed) as avg_tasks_completed,
    AVG(upm.deadline_success_rate) as avg_deadline_success,
    COUNT(DISTINCT upm.metric_date) as days_tracked
FROM user_productivity_metrics upm
GROUP BY upm.user_id, DATE_TRUNC('week', upm.metric_date);

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_productivity_summary_user_week 
    ON user_productivity_summary(user_id, week DESC);

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_productivity_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY user_productivity_summary;
END;
$$ LANGUAGE plpgsql;

-- Create trigger function for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to all tables with updated_at column
DO $$
DECLARE
    t text;
BEGIN
    FOR t IN 
        SELECT table_name 
        FROM information_schema.columns 
        WHERE column_name = 'updated_at' 
        AND table_schema = 'public'
    LOOP
        EXECUTE format('
            CREATE TRIGGER update_%I_updated_at 
            BEFORE UPDATE ON %I 
            FOR EACH ROW 
            EXECUTE FUNCTION update_updated_at_column();', 
            t, t
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Insert default AI preferences for new users
CREATE OR REPLACE FUNCTION create_default_ai_preferences()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO user_ai_preferences (
        user_id,
        learning_rate,
        feedback_sensitivity,
        privacy_level,
        enable_predictive_scheduling,
        enable_priority_suggestions,
        enable_time_estimates,
        enable_pattern_recognition,
        enable_proactive_insights,
        insight_frequency,
        prediction_confidence_threshold,
        suggestion_aggressiveness,
        prioritize_accuracy_over_speed,
        model_update_frequency
    ) VALUES (
        NEW.id,
        0.1,
        0.5,
        'balanced',
        true,
        true,
        true,
        true,
        true,
        'weekly',
        0.7,
        'moderate',
        false,
        'weekly'
    ) ON CONFLICT (user_id) DO NOTHING;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Note: This trigger would be created on the User table in the main database
-- For testing, you can manually insert preferences

-- Sample data for development/testing
INSERT INTO user_ai_preferences (
    id,
    user_id,
    learning_rate,
    feedback_sensitivity,
    privacy_level,
    enable_predictive_scheduling,
    enable_priority_suggestions,
    enable_time_estimates,
    enable_pattern_recognition,
    enable_proactive_insights
) VALUES (
    uuid_generate_v4(),
    'test-user-123',
    0.1,
    0.5,
    'balanced',
    true,
    true,
    true,
    true,
    true
) ON CONFLICT (user_id) DO NOTHING;