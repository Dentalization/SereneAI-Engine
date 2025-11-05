-- ============================================================================
-- SereneAI V2 Database Initialization
-- PostgreSQL schema for LangGraph checkpoints and application data
-- ============================================================================

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search

-- ============================================================================
-- LangGraph Checkpoints (managed by langgraph-checkpoint-postgres)
-- ============================================================================

-- The checkpoints table will be automatically created by AsyncPostgresSaver
-- This is just documentation of the expected schema:

-- CREATE TABLE IF NOT EXISTS checkpoints (
--     thread_id TEXT NOT NULL,
--     checkpoint_ns TEXT NOT NULL DEFAULT '',
--     checkpoint_id TEXT NOT NULL,
--     parent_checkpoint_id TEXT,
--     type TEXT,
--     checkpoint JSONB NOT NULL,
--     metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
--     created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
--     PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
-- );

-- CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id ON checkpoints(thread_id);
-- CREATE INDEX IF NOT EXISTS idx_checkpoints_parent ON checkpoints(parent_checkpoint_id);
-- CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at ON checkpoints(created_at DESC);

-- ============================================================================
-- Application Tables
-- ============================================================================

-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id TEXT UNIQUE,  -- External user ID from auth system
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_users_external_id ON users(external_id);
CREATE INDEX IF NOT EXISTS idx_users_last_active ON users(last_active_at DESC);

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    thread_id TEXT UNIQUE NOT NULL,
    stage TEXT NOT NULL DEFAULT 'greeting',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_thread_id ON conversations(thread_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_stage ON conversations(stage);

-- Messages table (for analytics and debugging)
CREATE TABLE IF NOT EXISTS messages (
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    role TEXT NOT NULL,  -- 'human', 'ai', 'system'
    content TEXT NOT NULL,
    image_url TEXT,
    sources JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);

-- User profiles (cached from state)
CREATE TABLE IF NOT EXISTS user_profiles (
    profile_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID UNIQUE REFERENCES users(user_id) ON DELETE CASCADE,
    demographics JSONB DEFAULT '{}'::jsonb,
    medical_history JSONB DEFAULT '{}'::jsonb,
    symptoms JSONB DEFAULT '{}'::jsonb,
    preferences JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);

-- Detection results (for analysis)
CREATE TABLE IF NOT EXISTS detection_results (
    detection_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    image_path TEXT NOT NULL,
    detections JSONB NOT NULL DEFAULT '[]'::jsonb,
    spatial_insights TEXT,
    quality_score FLOAT,
    quality_issues JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_detection_results_conversation_id ON detection_results(conversation_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_created_at ON detection_results(created_at DESC);

-- RAG queries (for analytics)
CREATE TABLE IF NOT EXISTS rag_queries (
    query_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    response TEXT,
    sources JSONB DEFAULT '[]'::jsonb,
    validation_result JSONB,
    execution_time_ms INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rag_queries_conversation_id ON rag_queries(conversation_id);
CREATE INDEX IF NOT EXISTS idx_rag_queries_created_at ON rag_queries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rag_queries_query_trgm ON rag_queries USING gin (query gin_trgm_ops);

-- Feedback table
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(conversation_id) ON DELETE SET NULL,
    message_id UUID REFERENCES messages(message_id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    rating INT CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_conversation_id ON feedback(conversation_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at DESC);

-- ============================================================================
-- Functions and Triggers
-- ============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE ON user_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Views for Analytics
-- ============================================================================

-- Active users view
CREATE OR REPLACE VIEW active_users AS
SELECT
    u.user_id,
    u.external_id,
    u.last_active_at,
    COUNT(DISTINCT c.conversation_id) as conversation_count,
    MAX(c.updated_at) as last_conversation_at
FROM users u
LEFT JOIN conversations c ON u.user_id = c.user_id
GROUP BY u.user_id, u.external_id, u.last_active_at;

-- Conversation statistics view
CREATE OR REPLACE VIEW conversation_stats AS
SELECT
    c.conversation_id,
    c.user_id,
    c.stage,
    c.created_at,
    c.updated_at,
    COUNT(DISTINCT m.message_id) as message_count,
    COUNT(DISTINCT d.detection_id) as detection_count,
    COUNT(DISTINCT r.query_id) as rag_query_count,
    EXTRACT(EPOCH FROM (c.updated_at - c.created_at)) as duration_seconds
FROM conversations c
LEFT JOIN messages m ON c.conversation_id = m.conversation_id
LEFT JOIN detection_results d ON c.conversation_id = d.conversation_id
LEFT JOIN rag_queries r ON c.conversation_id = r.conversation_id
GROUP BY c.conversation_id;

-- ============================================================================
-- Sample Data (Optional - for development)
-- ============================================================================

-- Insert a sample user (only in development)
-- INSERT INTO users (external_id, metadata) VALUES
--     ('dev_user_1', '{"env": "development"}'::jsonb)
-- ON CONFLICT (external_id) DO NOTHING;
