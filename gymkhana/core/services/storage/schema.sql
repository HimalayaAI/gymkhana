-- Gymkhana Core Database Schema V2
-- Complete rebuild with rollout tracking for GRPO training
-- Date: 2026-02-04
-- Tables ordered by insertion dependency

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- INDEPENDENT TABLES (No foreign key dependencies)
-- =============================================================================

-- Requests: Track LLM orchestrator calls
CREATE TABLE IF NOT EXISTS requests (
    id SERIAL PRIMARY KEY,
    model TEXT,
    messages JSONB,
    system TEXT,
    prompt_context_id TEXT,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    total_time FLOAT,
    max_tokens INTEGER,
    temperature FLOAT,
    raw_response JSONB,
    completion_tokens INTEGER,
    prompt_tokens INTEGER,
    total_tokens INTEGER,
    reasoning_tokens INTEGER,  -- NEW: Reasoning tokens for thinking models
    reasoning_content TEXT,  -- NEW: Reasoning/thinking content from models like o1, o3, DeepSeek R1, GLM-5
    raw_request JSONB,  -- Full API request payload (including tools) for debugging
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ShareGPT datasets: Post-processing for training data export
CREATE TABLE IF NOT EXISTS sharegpt_datasets (
    id SERIAL PRIMARY KEY,
    task_id TEXT,
    conversations JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- ROLLOUT TRACKING TABLES (Level 1 - Rollout Groups)
-- =============================================================================

-- Rollout Groups: Group all G parallel rollouts from the same task
CREATE TABLE IF NOT EXISTS rollout_groups (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id TEXT NOT NULL,
    environment TEXT NOT NULL,
    num_rollouts INTEGER NOT NULL,
    num_completed INTEGER DEFAULT 0,
    num_failed INTEGER DEFAULT 0,
    num_error INTEGER DEFAULT 0,
    num_timeout INTEGER DEFAULT 0,
    best_rollout_id UUID,  -- FK to rollouts table (set after rollouts complete)
    best_reward DOUBLE PRECISION DEFAULT 0.0,
    reward_mean DOUBLE PRECISION DEFAULT 0.0,
    reward_std DOUBLE PRECISION DEFAULT 0.0,
    reward_min DOUBLE PRECISION DEFAULT 0.0,
    config_json TEXT DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_rollout_groups_task_id ON rollout_groups(task_id);
CREATE INDEX IF NOT EXISTS idx_rollout_groups_environment ON rollout_groups(environment);
CREATE INDEX IF NOT EXISTS idx_rollout_groups_created_at ON rollout_groups(created_at);

-- =============================================================================
-- TRAJECTORY TABLES (Level 2 - Depends on rollout_groups)
-- =============================================================================

-- Trajectories: Top-level container for a rollout episode
CREATE TABLE IF NOT EXISTS trajectories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rollout_id UUID,  -- FK added later after rollouts table exists
    rollout_group_id UUID REFERENCES rollout_groups(id) ON DELETE SET NULL,
    rollout_index INTEGER,

    -- Task info
    task_id TEXT,
    environment TEXT,

    -- Interaction mode tracking (NEW)
    interaction_mode TEXT,  -- 'chat', 'tool_use', 'tool_use_interleaved', 'rlm'
    conversation_manager TEXT,  -- 'single_turn', 'multi_turn'
    max_turns INTEGER,  -- Maximum turns allowed for this trajectory

    -- Outcome
    success BOOLEAN NOT NULL,
    final_answer TEXT,
    expected_answer TEXT,
    answer_correct BOOLEAN,

    -- Execution metrics
    num_code_blocks INTEGER DEFAULT 0,
    num_errors INTEGER DEFAULT 0,
    num_turns INTEGER DEFAULT 0,  -- Actual turns taken

    -- Reward tracking
    total_reward DOUBLE PRECISION DEFAULT 0.0,
    step_rewards_json TEXT DEFAULT '[]',
    reward_function TEXT,

    -- Quality metrics
    efficiency_score DOUBLE PRECISION,
    quality_score DOUBLE PRECISION,

    -- Model config
    system_prompt TEXT,
    model_name TEXT,

    -- Metadata
    metadata_json TEXT DEFAULT '{}',

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_trajectories_rollout_group_id ON trajectories(rollout_group_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_task_id ON trajectories(task_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_environment ON trajectories(environment);
CREATE INDEX IF NOT EXISTS idx_trajectories_answer_correct ON trajectories(answer_correct);
CREATE INDEX IF NOT EXISTS idx_trajectories_total_reward ON trajectories(total_reward);
CREATE INDEX IF NOT EXISTS idx_trajectories_created_at ON trajectories(created_at);
CREATE INDEX IF NOT EXISTS idx_trajectories_interaction_mode ON trajectories(interaction_mode);
CREATE INDEX IF NOT EXISTS idx_trajectories_conversation_manager ON trajectories(conversation_manager);

-- =============================================================================
-- ROLLOUT TRACKING TABLES (Level 3 - Depends on trajectories)
-- =============================================================================

-- Rollouts: Individual rollout state with termination tracking
CREATE TABLE IF NOT EXISTS rollouts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rollout_group_id UUID NOT NULL REFERENCES rollout_groups(id) ON DELETE CASCADE,
    trajectory_id UUID REFERENCES trajectories(id) ON DELETE SET NULL,
    rollout_index INTEGER NOT NULL,

    -- Status tracking
    status TEXT NOT NULL,
    termination_reason TEXT,

    -- Execution metrics
    num_turns INTEGER DEFAULT 0,
    num_code_blocks INTEGER DEFAULT 0,
    num_errors INTEGER DEFAULT 0,
    consecutive_errors INTEGER DEFAULT 0,

    -- Format violation tracking
    num_format_violations INTEGER DEFAULT 0,
    consecutive_format_violations INTEGER DEFAULT 0,
    last_format_violation_type TEXT,
    format_violation_history_json TEXT DEFAULT '[]',

    -- Reward tracking
    total_reward DOUBLE PRECISION DEFAULT 0.0,

    -- Timing
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_ms DOUBLE PRECISION,

    -- Sandbox reference
    session_id TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_rollouts_group_id ON rollouts(rollout_group_id);
CREATE INDEX IF NOT EXISTS idx_rollouts_trajectory_id ON rollouts(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_rollouts_status ON rollouts(status);
CREATE INDEX IF NOT EXISTS idx_rollouts_total_reward ON rollouts(total_reward);
CREATE INDEX IF NOT EXISTS idx_rollouts_rollout_index ON rollouts(rollout_index);

-- Add foreign key constraint from trajectories to rollouts (circular reference resolved)
-- Note: This ALTER is only needed if table already exists. For fresh DB, constraint added inline above.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'fk_trajectories_rollout_id'
    ) THEN
        ALTER TABLE trajectories ADD CONSTRAINT fk_trajectories_rollout_id
            FOREIGN KEY (rollout_id) REFERENCES rollouts(id) ON DELETE SET NULL;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_trajectories_rollout_id ON trajectories(rollout_id);

-- =============================================================================
-- CONVERSATION TABLES (Level 4 - Depends on trajectories)
-- =============================================================================

-- Turns: Individual conversation turns within a trajectory
CREATE TABLE IF NOT EXISTS turns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trajectory_id UUID NOT NULL REFERENCES trajectories(id) ON DELETE CASCADE,
    turn_index INTEGER DEFAULT 0,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    code TEXT,
    reasoning_content TEXT,  -- NEW: Reasoning/thinking content from models like o1, o3, DeepSeek R1, GLM-5
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_turns_trajectory_id ON turns(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_turns_turn_index ON turns(turn_index);

-- =============================================================================
-- EXECUTION TABLES (Level 5 - Depends on turns)
-- =============================================================================

-- Executions: Detailed results of code execution in a turn
CREATE TABLE IF NOT EXISTS executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    turn_id UUID NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    success BOOLEAN NOT NULL,
    output TEXT DEFAULT '',
    error TEXT,
    truncated BOOLEAN DEFAULT FALSE,
    execution_time_ms INTEGER DEFAULT 0,
    files_created_json TEXT DEFAULT '[]',
    variables_json TEXT DEFAULT '{}',
    state_json TEXT DEFAULT '{}',
    state_formatted TEXT DEFAULT '(empty state)',
    done BOOLEAN DEFAULT FALSE,
    final_answer TEXT,
    iteration INTEGER DEFAULT 0,
    reward FLOAT DEFAULT 0.0,
    episode_state_json TEXT DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_executions_turn_id ON executions(turn_id);
CREATE INDEX IF NOT EXISTS idx_executions_success ON executions(success);

-- Sub-agent calls: Record sub-LLM invocations
CREATE TABLE IF NOT EXISTS sub_agent_calls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID REFERENCES executions(id) ON DELETE CASCADE,
    task TEXT NOT NULL,
    system_prompt TEXT,
    response TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sub_agent_calls_execution_id ON sub_agent_calls(execution_id);

-- =============================================================================
-- SANDBOX TABLES (Level 3 - Depends on rollout_groups, rollouts, trajectories)
-- =============================================================================

-- Sandbox sessions: Track container lifecycle with rollout references
CREATE TABLE IF NOT EXISTS sandbox_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id TEXT NOT NULL,
    rollout_id UUID REFERENCES rollouts(id) ON DELETE SET NULL,
    rollout_group_id UUID REFERENCES rollout_groups(id) ON DELETE SET NULL,
    trajectory_id UUID REFERENCES trajectories(id) ON DELETE SET NULL,
    environment TEXT,

    -- Status
    status TEXT NOT NULL,

    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ready_at TIMESTAMP WITH TIME ZONE,
    last_execution_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE,

    -- Execution stats
    total_reward DOUBLE PRECISION DEFAULT 0.0,
    total_executions INTEGER DEFAULT 0,
    successful_executions INTEGER DEFAULT 0,
    failed_executions INTEGER DEFAULT 0,
    total_execution_time_ms INTEGER DEFAULT 0,

    -- State snapshots
    interpreter_json TEXT DEFAULT '{}',
    episode_json TEXT DEFAULT '{}',
    config_json TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sandbox_sessions_session_id ON sandbox_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sandbox_sessions_rollout_id ON sandbox_sessions(rollout_id);
CREATE INDEX IF NOT EXISTS idx_sandbox_sessions_rollout_group_id ON sandbox_sessions(rollout_group_id);
CREATE INDEX IF NOT EXISTS idx_sandbox_sessions_trajectory_id ON sandbox_sessions(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_sandbox_sessions_environment ON sandbox_sessions(environment);

-- =============================================================================
-- CURATED VIEWS (Audit-friendly LLM trace views)
-- =============================================================================

-- LLM Requests: What was sent to the model
CREATE OR REPLACE VIEW llm_requests AS
SELECT
    id,
    model,
    system AS system_prompt,
    messages,
    raw_request -> 'tools' AS tools,
    CASE
        WHEN jsonb_typeof(raw_request -> 'tools') = 'array'
        THEN jsonb_array_length(raw_request -> 'tools')
        ELSE 0
    END AS num_tools,
    raw_request ->> 'tool_choice' AS tool_choice,
    temperature,
    max_tokens,
    created_at
FROM requests
ORDER BY id DESC;

-- LLM Responses: What came back from the model
CREATE OR REPLACE VIEW llm_responses AS
SELECT
    id,
    model,
    raw_response ->> 'content' AS content,
    raw_response -> 'raw_output' -> 'raw_result' -> 'choices' -> 0 -> 'message' -> 'tool_calls' AS tool_calls,
    CASE
        WHEN jsonb_typeof(raw_response -> 'raw_output' -> 'raw_result' -> 'choices' -> 0 -> 'message' -> 'tool_calls') = 'array'
        THEN jsonb_array_length(raw_response -> 'raw_output' -> 'raw_result' -> 'choices' -> 0 -> 'message' -> 'tool_calls')
        ELSE 0
    END AS num_tool_calls,
    raw_response -> 'raw_output' -> 'raw_result' -> 'choices' -> 0 ->> 'finish_reason' AS finish_reason,
    reasoning_content,
    prompt_tokens,
    completion_tokens,
    total_tokens,
    reasoning_tokens,
    raw_response ->> 'time_taken' AS time_taken_sec,
    created_at
FROM requests
ORDER BY id DESC;
