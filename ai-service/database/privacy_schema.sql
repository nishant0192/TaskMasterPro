-- ai-service/database/privacy_schema.sql
-- Privacy-focused database schema for training data

-- User privacy settings and consent management
CREATE TABLE user_privacy_settings (
    user_id UUID PRIMARY KEY,
    ai_training_consent BOOLEAN DEFAULT FALSE,
    federated_learning_consent BOOLEAN DEFAULT FALSE,
    data_retention_days INTEGER DEFAULT 365,
    anonymization_preference VARCHAR(20) DEFAULT 'full',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Privacy-protected training data storage
CREATE TABLE user_training_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    task_id UUID NOT NULL,
    
    -- Hashed/anonymized content (no raw text)
    title_hash VARCHAR(32),
    description_hash VARCHAR(32),
    category VARCHAR(50),
    
    -- Numerical features (less sensitive)
    estimated_duration INTEGER,
    actual_duration INTEGER,
    priority INTEGER,
    ai_suggested_priority INTEGER,
    completion_status VARCHAR(20),
    user_satisfaction INTEGER,
    
    -- Temporal features
    created_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Privacy controls
    consent_for_training BOOLEAN DEFAULT FALSE,
    anonymization_level VARCHAR(20) DEFAULT 'standard',
    retention_expires_at TIMESTAMP,
    
    -- Constraints
    UNIQUE(user_id, task_id),
    FOREIGN KEY (user_id) REFERENCES user_privacy_settings(user_id)
);

-- Audit trail for data access
CREATE TABLE privacy_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    operation VARCHAR(50),  -- 'training_data_access', 'model_training', etc.
    data_accessed TEXT,     -- Description of what data was accessed
    purpose VARCHAR(100),   -- 'model_training', 'inference', etc.
    accessed_by VARCHAR(100), -- Service or user who accessed
    access_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    privacy_controls_applied JSONB
);

-- Anonymized global statistics (for global model training)
CREATE TABLE anonymized_task_statistics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Aggregated, non-identifiable statistics
    task_category VARCHAR(50),
    duration_bucket VARCHAR(20),  -- '0-30min', '30-60min', etc.
    priority_level INTEGER,
    completion_rate DECIMAL(3,2),
    average_satisfaction DECIMAL(3,2),
    
    -- Time-based aggregation
    time_period VARCHAR(20),      -- 'morning', 'afternoon', etc.
    day_type VARCHAR(20),         -- 'weekday', 'weekend'
    
    -- Statistical metadata
    sample_size INTEGER,          -- How many users contributed
    confidence_interval DECIMAL(3,2),
    noise_applied BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient privacy-aware queries
CREATE INDEX idx_training_data_user_consent ON user_training_data(user_id, consent_for_training);
CREATE INDEX idx_training_data_retention ON user_training_data(retention_expires_at);
CREATE INDEX idx_privacy_audit_user ON privacy_audit_log(user_id, access_timestamp);