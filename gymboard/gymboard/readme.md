# Gymkhana Dashboard

A web-based dashboard for monitoring and analyzing Gymkhana RL training data generation. Provides real-time visualization of trajectories, rollouts, and training data quality with specialized features for LLM agent conversations.

## Overview

The dashboard connects to the Gymkhana PostgreSQL database to provide insights into:
- **Trajectories**: Full conversation histories with code execution
- **Rollouts**: Parallel solution attempts with reward tracking
- **ShareGPT Datasets**: Validated training data ready for export
- **Executions**: Individual code block runs with outputs
- **Requests**: LLM API calls with token usage

## Key Features

### LLM-Specific Visualizations

**Conversational View:**
- Bubble chat interface with role-based styling
- Roles: System (crimson), User/Human (green), Assistant/GPT (blue), Tool (gray)
- XML tag highlighting for `<python>`, `<bash>`, `<think>`, `<final_answer>`
- Collapsible view with message count preview
- Inline expansion for quick inspection

**JSON View:**
- Syntax-highlighted JSON with proper formatting
- Nested structure visualization
- XML tag preservation in JSON strings
- Expandable/collapsible sections

**Text View:**
- Raw text fallback for non-JSON content
- Preserves formatting and whitespace
- HTML escaping for safety

### Data Exploration

**Table Browser:**
- View all database tables (trajectories, rollouts, sharegpt_datasets, etc.)
- Automatic column detection including JSON fields
- Pagination with configurable page sizes (10-1000 rows)
- Column sorting and filtering
- UUID truncation for cleaner display

**JSON Path Querying:**
- Query nested JSON fields directly: `metadata_json.task_id`
- Array indexing: `conversations_json[0].value`
- Automatic type detection (text, numeric, timestamp, boolean)
- Dynamic schema inference from sample data

**Full-Text Search:**
- Search across all columns in a table
- Type-aware searching (dates, numbers, text, JSON)
- Case-insensitive matching
- Pagination of search results

### Chart Visualization

**Interactive Plotly Charts:**
- Scatter plots for numerical data
- Bar charts for categorical data
- Time series for temporal data
- Automatic chart type selection
- Hover tooltips with details
- Export to PNG

**Smart Data Analysis:**
- Automatic data type detection
- Distribution pattern analysis
- Relationship visualization
- Responsive design

## Quick Start

### Starting the Dashboard

**From project root:**
```bash
# Start dashboard on port 8000
python -m gymkhana.dashboard.app

# Or with uvicorn directly
uvicorn gymkhana.dashboard.app:app --host 0.0.0.0 --port 8000
```

**Access at:** `http://localhost:8000`

### Configuration

The dashboard reads database credentials from environment variables:

```bash
# .env file
DB_NAME=gymkhana
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

# Optional: Flatten nested JSON in responses
DASHBOARD_FLATTEN_JSON=false
```

## Common Use Cases

### Monitoring Training Runs

**View recent trajectories:**
1. Select `trajectories` table
2. Sort by `created_at` descending
3. Click conversation to see full chat history
4. Check `success` and `final_answer` fields

**Analyze rollout performance:**
1. Select `rollout_groups` table
2. View `mean_reward`, `best_reward`, `num_rollouts`
3. Join with `rollouts` to see individual attempts
4. Filter by `task_id` to compare rollouts for same task

**Inspect ShareGPT quality:**
1. Select `sharegpt_datasets` table
2. View `conversations_json` in conversational mode
3. Check for proper format (no consecutive same-role messages)
4. Verify `<final_answer>` tags present

### Debugging Failed Rollouts

**Find failed trajectories:**
```sql
-- Use search feature
Table: trajectories
Search: "Error during rollout"
```

**Check termination reasons:**
1. Select `rollouts` table
2. Filter by `status = 'failed'`
3. View `termination_reason` field
4. Analyze patterns (max_turns, format_violations, etc.)

**Inspect execution errors:**
1. Select `executions` table
2. Filter by `success = false`
3. View `error` field for stack traces
4. Check `code` field to see what failed

### Analyzing Model Behavior

**View code execution patterns:**
1. Select `executions` table
2. Group by `language` (python vs bash)
3. Analyze `execution_time_ms` distribution
4. Check `output` for common patterns

**Track token usage:**
1. Select `requests` table
2. View `prompt_tokens`, `completion_tokens`, `total_tokens`
3. Calculate costs based on model pricing
4. Identify expensive queries

**Compare model responses:**
1. Select `trajectories` table
2. Filter by same `task_id`
3. View different rollout attempts
4. Compare approaches and success rates

## Database Schema

### Core Tables

**trajectories:**
- `id`: UUID primary key
- `task_id`: Task identifier
- `turns_json`: Full conversation history
- `success`: Boolean completion status
- `final_answer`: Extracted answer text
- `metadata_json`: Task metadata
- `created_at`: Timestamp

**rollout_groups:**
- `id`: UUID primary key
- `task_id`: Task identifier
- `num_rollouts`: Number of parallel attempts
- `mean_reward`: Average reward across rollouts
- `best_reward`: Best rollout reward
- `std_reward`: Standard deviation
- `created_at`: Timestamp

**rollouts:**
- `id`: UUID primary key
- `group_id`: Foreign key to rollout_groups
- `rollout_index`: Index within group (0 to N-1)
- `trajectory_id`: Foreign key to trajectories
- `reward`: Rollout reward
- `status`: completed/failed/error/timeout
- `termination_reason`: Why rollout ended
- `answer_correctness`: correct/incorrect/unknown

**sharegpt_datasets:**
- `id`: UUID primary key
- `trajectory_id`: Foreign key to trajectories
- `conversations_json`: ShareGPT format conversation
- `metadata_json`: Task and model metadata
- `created_at`: Timestamp

**executions:**
- `id`: UUID primary key
- `turn_id`: Foreign key to turn
- `code`: Code that was executed
- `language`: python/bash
- `output`: Execution output
- `error`: Error message if failed
- `success`: Boolean
- `execution_time_ms`: Duration

**requests:**
- `id`: UUID primary key
- `prompt_tokens`: Input tokens
- `completion_tokens`: Output tokens
- `total_tokens`: Sum
- `model`: Model name
- `created_at`: Timestamp

## API Reference

### Endpoints

**GET /api/get-tables**
- Returns: List of non-empty tables
- Example: `["trajectories", "rollouts", "sharegpt_datasets"]`

**GET /api/column-names?table_name={table}**
- Returns: Column metadata including JSON paths
- Example: `[{"name": "id", "type": "uuid"}, {"name": "metadata_json.task_id", "type": "text"}]`

**GET /api/metrics-data**
- Parameters:
  - `table_name`: Table to query
  - `full_table`: Boolean (true for all columns)
  - `x_column`, `y_column`: For charts
  - `page`, `page_size`: Pagination
- Returns: Paginated data with total count

**GET /api/search**
- Parameters:
  - `table_name`: Table to search
  - `search_term`: Query string
  - `columns`: Optional column list
  - `page`, `page_size`: Pagination
- Returns: Matching rows with total count

### JSON Path Syntax

**Simple field:**
```
metadata_json.task_id
```

**Nested field:**
```
metadata_json.extra.repo
```

**Array element:**
```
conversations_json[0].value
```

**Nested array:**
```
turns_json[0].executions[0].output
```

## Tech Stack

**Backend:**
- FastAPI for REST API
- Psycopg2 for PostgreSQL connectivity
- Python-dotenv for configuration
- Direct SQL queries (no ORM) for performance

**Frontend:**
- Vanilla JavaScript (no framework)
- Plotly.js for charts
- Modern CSS with flexbox/grid
- Responsive design

**Database:**
- PostgreSQL 13+
- JSON/JSONB support
- Efficient indexing on common queries

## Performance

**Optimizations:**
- Pagination for large result sets
- JSON path queries use PostgreSQL operators (`->`, `->>`)
- Column type caching
- Debounced search (300ms)
- Lazy loading of conversation views

**Limits:**
- Max page size: 1000 rows
- Search timeout: 30 seconds
- JSON sample size: 100 rows for schema detection

## Security

**SQL Injection Prevention:**
- All queries use parameterized statements
- `psycopg2.sql` module for safe identifier composition
- No string concatenation for SQL

**Data Sanitization:**
- HTML escaping in frontend
- JSON validation before parsing
- Error message sanitization

**Access Control:**
- Dashboard assumes trusted network
- No authentication (add reverse proxy for production)
- Read-only database user recommended

## Troubleshooting

**Dashboard won't start:**
```bash
# Check database connection
psql -h localhost -U postgres -d gymkhana

# Verify environment variables
echo $DB_NAME $DB_USER $DB_HOST $DB_PORT

# Check port availability
lsof -i :8000
```

**No tables showing:**
```bash
# Verify tables exist and have data
psql -d gymkhana -c "\dt"
psql -d gymkhana -c "SELECT COUNT(*) FROM trajectories;"
```

**JSON paths not working:**
```bash
# Verify PostgreSQL version (need 9.3+)
psql -d gymkhana -c "SELECT version();"

# Test JSON query manually
psql -d gymkhana -c "SELECT metadata_json->>'task_id' FROM trajectories LIMIT 1;"
```

**Slow queries:**
```sql
-- Add indexes for common queries
CREATE INDEX idx_trajectories_task_id ON trajectories((metadata_json->>'task_id'));
CREATE INDEX idx_trajectories_created_at ON trajectories(created_at DESC);
CREATE INDEX idx_rollouts_group_id ON rollouts(group_id);
```

## Directory Structure

```
gymkhana/core/dashboard/
├── app.py              # FastAPI application
├── static/
│   ├── index.html      # Single-page application
│   ├── styles.css      # Dashboard styling
│   ├── marketagents_logo.png
│   └── marketagents_logo.jpg
└── readme.md           # This file
```

## Future Enhancements

- Real-time updates via WebSockets
- Custom dashboard layouts
- Export to CSV/Excel
- Advanced filtering UI
- Rollout comparison view
- Token cost calculator
- Model performance analytics
- A/B test visualization
