-- ai-service/database/schema.sql

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- User behavior patterns tracking
CREATE TABLE user_behavior_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    pattern_type VARCHAR(100) NOT NULL, -- 'productivity_hours', 'task_preferences', 'completion_patterns'
    pattern_data JSONB NOT NULL,
    confidence_score FLOAT DEFAULT 0.5,
    frequency_count INTEGER DEFAULT 1,
    last_observed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1)
);

-- Indexes for behavior patterns
CREATE INDEX idx_behavior_patterns_user_type ON user_behavior_patterns(user_id, pattern_type);
CREATE INDEX idx_behavior_patterns_confidence ON user_behavior_patterns(confidence_score DESC);
CREATE INDEX idx_behavior_patterns_frequency ON user_behavior_patterns(frequency_count DESC);

-- Personal AI model weights and metadata
CREATE TABLE user_model_weights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'prioritization', 'scheduling', 'nlp', 'prediction'
    model_subtype VARCHAR(50), -- 'classification', 'regression', 'embedding'
    weights BYTEA NOT NULL, -- Compressed serialized model weights
    metadata JSONB, -- Model parameters, architecture, etc.
    accuracy_score FLOAT,
    training_samples INTEGER DEFAULT 0,
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, model_type, model_subtype, version)
);

-- Indexes for model weights
CREATE INDEX idx_model_weights_user_active ON user_model_weights(user_id, model_type, is_active);
CREATE INDEX idx_model_weights_accuracy ON user_model_weights(accuracy_score DESC);

-- Training sessions and feedback
CREATE TABLE ai_training_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    session_type VARCHAR(50) NOT NULL, -- 'initial', 'incremental', 'feedback'
    model_type VARCHAR(50) NOT NULL,
    training_data JSONB NOT NULL,
    feedback_data JSONB,
    performance_metrics JSONB,
    before_accuracy FLOAT,
    after_accuracy FLOAT,
    training_duration_ms INTEGER,
    sample_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for training sessions
CREATE INDEX idx_training_sessions_user_model ON ai_training_sessions(user_id, model_type);
CREATE INDEX idx_training_sessions_type ON ai_training_sessions(session_type);
CREATE INDEX idx_training_sessions_performance ON ai_training_sessions(after_accuracy DESC);

-- Task embeddings for semantic similarity
CREATE TABLE task_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    task_id UUID NOT NULL,
    embedding vector(384) NOT NULL, -- Using all-MiniLM-L6-v2 (384 dimensions)
    title_embedding vector(384),
    description_embedding vector(384),
    
    -- Task metadata for feature learning
    category VARCHAR(100),
    priority INTEGER,
    estimated_duration INTEGER, -- minutes
    actual_duration INTEGER, -- minutes
    completion_status VARCHAR(20), -- 'completed', 'abandoned', 'delayed'
    completion_quality FLOAT, -- 0-1 score of how well task was completed
    
    -- Context when task was created/completed
    created_hour INTEGER, -- 0-23
    created_day_of_week INTEGER, -- 0-6
    completed_hour INTEGER,
    user_energy_level FLOAT, -- 0-1 estimated energy when task was created
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_priority CHECK (priority >= 1 AND priority <= 5),
    CONSTRAINT valid_energy CHECK (user_energy_level >= 0 AND user_energy_level <= 1)
);

-- Indexes for task embeddings (vector similarity search)
CREATE INDEX ON task_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON task_embeddings USING ivfflat (title_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_task_embeddings_user ON task_embeddings(user_id);
CREATE INDEX idx_task_embeddings_category ON task_embeddings(user_id, category);
CREATE INDEX idx_task_embeddings_completion ON task_embeddings(completion_status, completion_quality DESC);

-- User preference embeddings
CREATE TABLE user_preference_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    preference_type VARCHAR(50) NOT NULL, -- 'task_style', 'time_preference', 'priority_style'
    preference_context VARCHAR(100), -- 'work', 'personal', 'health', etc.
    embedding vector(384) NOT NULL,
    weight FLOAT DEFAULT 1.0, -- Importance weight
    confidence FLOAT DEFAULT 0.5,
    usage_count INTEGER DEFAULT 1,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_weight CHECK (weight >= 0),
    CONSTRAINT valid_confidence_pref CHECK (confidence >= 0 AND confidence <= 1)
);

-- Indexes for preference embeddings
CREATE INDEX ON user_preference_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX idx_preference_embeddings_user_type ON user_preference_embeddings(user_id, preference_type);
CREATE INDEX idx_preference_embeddings_weight ON user_preference_embeddings(weight DESC);

-- Personal vocabulary and language patterns
CREATE TABLE user_vocabulary (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    term VARCHAR(200) NOT NULL,
    normalized_form VARCHAR(200) NOT NULL,
    term_type VARCHAR(50) NOT NULL, -- 'abbreviation', 'synonym', 'category', 'person', 'location'
    context VARCHAR(100), -- Additional context about when this term is used
    frequency INTEGER DEFAULT 1,
    confidence FLOAT DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, term, term_type)
);

-- Indexes for user vocabulary
CREATE INDEX idx_vocabulary_user_term ON user_vocabulary(user_id, term);
CREATE INDEX idx_vocabulary_frequency ON user_vocabulary(frequency DESC);
CREATE INDEX idx_vocabulary_type ON user_vocabulary(term_type);

-- Productivity analytics and insights
CREATE TABLE user_productivity_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    metric_date DATE NOT NULL,
    
    -- Core productivity metrics
    tasks_created INTEGER DEFAULT 0,
    tasks_completed INTEGER DEFAULT 0,
    tasks_abandoned INTEGER DEFAULT 0,
    avg_completion_time FLOAT, -- minutes
    productivity_score FLOAT, -- 0-1 composite score
    
    -- Time-based metrics
    peak_productivity_hours INTEGER[], -- Array of hours (0-23)
    total_focus_time INTEGER, -- minutes
    break_frequency FLOAT, -- breaks per hour
    
    -- Quality metrics
    deadline_success_rate FLOAT, -- 0-1
    task_quality_score FLOAT, -- 0-1
    priority_accuracy FLOAT, -- How accurate user's initial priority was
    
    -- Behavioral patterns
    procrastination_score FLOAT, -- 0-1, higher = more procrastination
    consistency_score FLOAT, -- 0-1, how consistent user is
    stress_indicators JSONB, -- Various stress-related metrics
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, metric_date)
);

-- Indexes for productivity metrics
CREATE INDEX idx_productivity_metrics_user_date ON user_productivity_metrics(user_id, metric_date DESC);
CREATE INDEX idx_productivity_metrics_score ON user_productivity_metrics(productivity_score DESC);

-- AI predictions and their outcomes
CREATE TABLE ai_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    prediction_type VARCHAR(50) NOT NULL, -- 'completion_time', 'priority', 'success_probability'
    target_task_id UUID,
    
    -- Prediction details
    predicted_value FLOAT NOT NULL,
    predicted_confidence FLOAT NOT NULL,
    prediction_context JSONB, -- Context used to make prediction
    
    -- Actual outcome (filled in later)
    actual_value FLOAT,
    prediction_accuracy FLOAT, -- How accurate was this prediction
    feedback_provided BOOLEAN DEFAULT false,
    user_satisfaction INTEGER, -- 1-5 rating of prediction usefulness
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    outcome_recorded_at TIMESTAMP,
    
    CONSTRAINT valid_confidence_pred CHECK (predicted_confidence >= 0 AND predicted_confidence <= 1)
);

-- Indexes for AI predictions
CREATE INDEX idx_predictions_user_type ON ai_predictions(user_id, prediction_type);
CREATE INDEX idx_predictions_accuracy ON ai_predictions(prediction_accuracy DESC) WHERE prediction_accuracy IS NOT NULL;
CREATE INDEX idx_predictions_satisfaction ON ai_predictions(user_satisfaction DESC) WHERE user_satisfaction IS NOT NULL;

-- Feature importance tracking
CREATE TABLE feature_importance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    importance_score FLOAT NOT NULL,
    model_version INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, model_type, feature_name, model_version)
);

-- Indexes for feature importance
CREATE INDEX idx_feature_importance_user_model ON feature_importance(user_id, model_type);
CREATE INDEX idx_feature_importance_score ON feature_importance(importance_score DESC);

-- User AI preferences and settings
CREATE TABLE user_ai_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL UNIQUE,
    
    -- Learning preferences
    learning_rate FLOAT DEFAULT 0.1, -- How quickly to adapt to new patterns
    feedback_sensitivity FLOAT DEFAULT 0.5, -- How much to weight user feedback
    privacy_level VARCHAR(20) DEFAULT 'balanced', -- 'minimal', 'balanced', 'full'
    
    -- Feature preferences
    enable_predictive_scheduling BOOLEAN DEFAULT true,
    enable_priority_suggestions BOOLEAN DEFAULT true,
    enable_time_estimates BOOLEAN DEFAULT true,
    enable_pattern_recognition BOOLEAN DEFAULT true,
    enable_proactive_insights BOOLEAN DEFAULT true,
    
    -- Notification preferences for AI
    insight_frequency VARCHAR(20) DEFAULT 'weekly', -- 'daily', 'weekly', 'monthly'
    prediction_confidence_threshold FLOAT DEFAULT 0.7,
    suggestion_aggressiveness VARCHAR(20) DEFAULT 'moderate', -- 'conservative', 'moderate', 'aggressive'
    
    -- Model preferences
    prioritize_accuracy_over_speed BOOLEAN DEFAULT false,
    model_update_frequency VARCHAR(20) DEFAULT 'weekly', -- 'daily', 'weekly', 'monthly'
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_learning_rate CHECK (learning_rate > 0 AND learning_rate <= 1),
    CONSTRAINT valid_feedback_sensitivity CHECK (feedback_sensitivity >= 0 AND feedback_sensitivity <= 1),
    CONSTRAINT valid_confidence_threshold CHECK (prediction_confidence_threshold >= 0 AND prediction_confidence_threshold <= 1)
);

-- Model performance tracking
CREATE TABLE model_performance_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    model_version INTEGER NOT NULL,
    
    -- Performance metrics
    accuracy FLOAT,
    precision_score FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    mean_absolute_error FLOAT,
    user_satisfaction_avg FLOAT,
    
    -- Usage statistics
    predictions_made INTEGER DEFAULT 0,
    feedback_received INTEGER DEFAULT 0,
    positive_feedback_ratio FLOAT,
    
    -- Context
    evaluation_date DATE NOT NULL,
    sample_size INTEGER,
    test_conditions JSONB,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, model_type, model_version, evaluation_date)
);

-- Indexes for model performance
CREATE INDEX idx_model_performance_user_model ON model_performance_history(user_id, model_type);
CREATE INDEX idx_model_performance_date ON model_performance_history(evaluation_date DESC);
CREATE INDEX idx_model_performance_accuracy ON model_performance_history(accuracy DESC) WHERE accuracy IS NOT NULL;

-- User interaction logs for continuous learning
CREATE TABLE user_interaction_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    interaction_type VARCHAR(50) NOT NULL, -- 'task_created', 'priority_changed', 'task_completed', 'suggestion_accepted'
    
    -- Context of interaction
    task_id UUID,
    before_state JSONB,
    after_state JSONB,
    user_context JSONB, -- Time, location, calendar state, etc.
    
    -- AI involvement
    ai_suggestion_provided BOOLEAN DEFAULT false,
    ai_prediction_made BOOLEAN DEFAULT false,
    suggestion_accepted BOOLEAN,
    prediction_accuracy FLOAT,
    
    -- Timing
    interaction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id UUID, -- Group related interactions
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for interaction logs
CREATE INDEX idx_interaction_logs_user_type ON user_interaction_logs(user_id, interaction_type);
CREATE INDEX idx_interaction_logs_timestamp ON user_interaction_logs(interaction_timestamp DESC);
CREATE INDEX idx_interaction_logs_session ON user_interaction_logs(session_id);
CREATE INDEX idx_interaction_logs_ai_involvement ON user_interaction_logs(ai_suggestion_provided, suggestion_accepted);

-- Cached embeddings for performance
CREATE TABLE embedding_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_hash VARCHAR(64) NOT NULL UNIQUE, -- SHA-256 hash of content
    content_type VARCHAR(50) NOT NULL, -- 'task_title', 'task_description', 'user_query'
    embedding vector(384) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    usage_count INTEGER DEFAULT 1,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for embedding cache
CREATE INDEX ON embedding_cache USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_embedding_cache_hash ON embedding_cache(content_hash);
CREATE INDEX idx_embedding_cache_usage ON embedding_cache(usage_count DESC, last_used DESC);

-- AI model registry (for version control and rollback)
CREATE TABLE ai_model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'global', 'user_specific'
    
    -- Model metadata
    description TEXT,
    architecture JSONB, -- Model architecture details
    hyperparameters JSONB,
    training_config JSONB,
    
    -- Performance benchmarks
    benchmark_scores JSONB,
    validation_metrics JSONB,
    
    -- Deployment info
    is_active BOOLEAN DEFAULT false,
    deployment_date TIMESTAMP,
    deprecated_date TIMESTAMP,
    
    -- File references
    model_file_path VARCHAR(500),
    checkpoint_path VARCHAR(500),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_name, model_version)
);

-- Indexes for model registry
CREATE INDEX idx_model_registry_active ON ai_model_registry(model_name, is_active);
CREATE INDEX idx_model_registry_type ON ai_model_registry(model_type);

-- Functions for automatic updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_user_behavior_patterns_updated_at BEFORE UPDATE ON user_behavior_patterns FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_model_weights_updated_at BEFORE UPDATE ON user_model_weights FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_task_embeddings_updated_at BEFORE UPDATE ON task_embeddings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_vocabulary_updated_at BEFORE UPDATE ON user_vocabulary FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_ai_preferences_updated_at BEFORE UPDATE ON user_ai_preferences FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_ai_model_registry_updated_at BEFORE UPDATE ON ai_model_registry FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Views for common queries
CREATE VIEW user_ai_summary AS
SELECT 
    u.user_id,
    COUNT(DISTINCT ubp.id) as behavior_patterns_count,
    COUNT(DISTINCT umw.id) as personal_models_count,
    COUNT(DISTINCT te.id) as task_embeddings_count,
    COUNT(DISTINCT uv.id) as vocabulary_terms_count,
    AVG(te.completion_quality) as avg_task_quality,
    MAX(umw.updated_at) as last_model_update,
    MAX(ubp.updated_at) as last_pattern_update
FROM (SELECT DISTINCT user_id FROM user_behavior_patterns) u
LEFT JOIN user_behavior_patterns ubp ON u.user_id = ubp.user_id
LEFT JOIN user_model_weights umw ON u.user_id = umw.user_id AND umw.is_active = true
LEFT JOIN task_embeddings te ON u.user_id = te.user_id
LEFT JOIN user_vocabulary uv ON u.user_id = uv.user_id
GROUP BY u.user_id;

-- Performance monitoring view
CREATE VIEW model_performance_summary AS
SELECT 
    user_id,
    model_type,
    COUNT(*) as total_predictions,
    AVG(prediction_accuracy) as avg_accuracy,
    AVG(user_satisfaction) as avg_satisfaction,
    COUNT(CASE WHEN feedback_provided THEN 1 END) as feedback_count,
    MAX(created_at) as last_prediction
FROM ai_predictions 
WHERE prediction_accuracy IS NOT NULL
GROUP BY user_id, model_type;

-- Cleanup function for old data
CREATE OR REPLACE FUNCTION cleanup_old_ai_data()
RETURNS void AS $
BEGIN
    -- Delete old embedding cache entries (older than 30 days, low usage)
    DELETE FROM embedding_cache 
    WHERE last_used < CURRENT_TIMESTAMP - INTERVAL '30 days' 
    AND usage_count < 5;
    
    -- Delete old interaction logs (older than 90 days)
    DELETE FROM user_interaction_logs 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '90 days';
    
    -- Delete old training sessions (keep only last 50 per user per model type)
    DELETE FROM ai_training_sessions
    WHERE id IN (
        SELECT id FROM (
            SELECT id, 
                   ROW_NUMBER() OVER (PARTITION BY user_id, model_type ORDER BY created_at DESC) as rn
            FROM ai_training_sessions
        ) ranked
        WHERE rn > 50
    );
    
    -- Archive old predictions (move to separate table if needed)
    -- This is left as a placeholder for your archival strategy
    
END;
$ LANGUAGE plpgsql;

-- Create a scheduled job hint (implement with pg_cron or external scheduler)
-- SELECT cron.schedule('cleanup-ai-data', '0 2 * * 0', 'SELECT cleanup_old_ai_data();');

COMMENT ON TABLE user_behavior_patterns IS 'Stores learned behavioral patterns for each user';
COMMENT ON TABLE user_model_weights IS 'Stores personalized AI model weights for each user';
COMMENT ON TABLE task_embeddings IS 'Vector embeddings of tasks for semantic similarity search';
COMMENT ON TABLE user_vocabulary IS 'Personal vocabulary and language patterns for each user';
COMMENT ON TABLE ai_predictions IS 'AI predictions and their actual outcomes for accuracy tracking';
COMMENT ON TABLE model_performance_history IS 'Historical performance metrics for AI models';
COMMENT ON FUNCTION cleanup_old_ai_data() IS 'Maintenance function to clean up old AI data and prevent database bloat';