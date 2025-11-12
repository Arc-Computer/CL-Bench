# Token Tracking Investigation Report

## Executive Summary

Investigation into why student tokens and reward system tokens are not being extracted and stored in `atlas_metadata.token_usage`. Found that:

1. **Student tokens** ARE being tracked by Atlas SDK in `ExecutionContext.metadata['token_usage']` (flat structure)
2. **Reward tokens** ARE being tracked in `ExecutionContext.metadata['session_reward_audit'][].raw_response.usage`
3. **Learning tokens** ARE being tracked in `ExecutionContext.metadata['token_usage']['learning']`
4. **Judge tokens** ARE being tracked in `conversation_result.per_turn_results[].token_usage`

However, these tokens are NOT appearing in `atlas_metadata.token_usage` in the final session results.

## Architecture Overview

### ExecutionContext Structure

- Uses `ContextVar` (thread-local/async-context-local storage)
- `metadata` is a dict stored in `ContextVar`
- `reset()` clears the metadata dict
- `get()` returns `ExecutionContext` wrapper around singleton `ExecutionContextState`
- Each async context has its own `ContextVar` instance

### Token Storage Locations

#### 1. Student Tokens
- **Location**: `ExecutionContext.metadata['token_usage']` (FLAT structure)
- **Structure**:
  ```python
  {
      'prompt_tokens': int,
      'completion_tokens': int,
      'total_tokens': int,
      'calls': int
  }
  ```
- **Populated by**: `student.py._apply_usage_payload()` method
- **Called from**:
  - `acreate_plan()` line 189: `self._apply_usage_payload(getattr(response, "usage", None))`
  - `aexecute_step()` line 314: `self._apply_usage_payload(usage_metadata)`
  - `asynthesize_final_answer()` line 361: `self._apply_usage_payload(getattr(response, "usage", None))`
  - Stream events line 973: `usage_snapshot = self._apply_usage_payload(usage_payload)`
- **Source**: LiteLLM response `usage` field from adapter calls

#### 2. Reward Tokens
- **Location**: `ExecutionContext.metadata['session_reward_audit'][].raw_response.usage`
- **Structure**:
  ```python
  {
      'prompt_tokens': int,
      'completion_tokens': int,
      'total_tokens': int
  }
  ```
- **Populated by**: Teacher `LLMClient` calls (`teacher.py`)
- **Stored via**: `ExecutionContext.set_session_reward(audit=...)` in reward system
- **Source**: LiteLLM `raw_response.usage` from teacher validation/review calls

#### 3. Learning Tokens
- **Location**: `ExecutionContext.metadata['token_usage']['learning']`
- **Structure**:
  ```python
  {
      'prompt_tokens': int,
      'completion_tokens': int,
      'total_tokens': int
  }
  ```
- **Populated by**: `synthesizer.py` lines 148-166
- **Extracted from**: `LLMResponse.raw.usage` after learning synthesis

#### 4. Judge Tokens
- **Location**: `conversation_result.per_turn_results[].token_usage`
- **Structure**:
  ```python
  {
      'judge': {
          'prompt_tokens': int,
          'completion_tokens': int,
          'total_tokens': int
      },
      'judge_response': {
          'prompt_tokens': int,
          'completion_tokens': int,
          'total_tokens': int
      }
  }
  ```
- **Populated by**: `conversation_harness.py` (lines 784, 821, 850)
- **Stored in**: `per_turn_results` (NOT in ExecutionContext)

## Current Extraction Flow

### In `atlas_integration.py`

1. `_run_single_task()`:
   - Line 264: `ExecutionContext.reset()` - clears metadata
   - Line 287: `atlas_arun()` - executes conversation (populates ExecutionContext)
   - Line 298: `_store_student_and_judge_token_usage()` - extracts from `conversation_result`
   - Line 300: `_extract_atlas_metadata()` - reads from ExecutionContext

2. `_store_student_and_judge_token_usage()`:
   - Extracts student tokens from `conversation_result.metadata.agent.token_usage`
   - Extracts judge tokens from `per_turn_results[].token_usage`
   - Stores in `ExecutionContext.metadata['token_usage']` under `student` and `judge` keys

3. `_extract_atlas_metadata()`:
   - Reads `ExecutionContext.metadata`
   - Extracts `token_usage` key (line 208)
   - Returns snapshot of metadata

## Issues Identified

### Issue 1: Student Tokens Not Extracted from ExecutionContext

**Problem**: Student tokens are stored in `ExecutionContext.metadata['token_usage']` as a FLAT structure (with `calls` key), but we're trying to extract them from `conversation_result.metadata.agent.token_usage` which may be empty.

**Root Cause**: 
- Atlas SDK stores student tokens directly in ExecutionContext during execution
- Our extraction function looks in the wrong place (`conversation_result` instead of ExecutionContext)
- The flat structure needs to be moved to `token_usage['student']`

**Evidence**:
- `student.py._apply_usage_payload()` stores tokens in `context.metadata.setdefault("token_usage", {...})`
- This creates a flat structure, not nested under `student` key
- `_aggregate_token_usage()` in `conversation_harness.py` aggregates from `per_turn_results`, but student tokens from Atlas SDK are NOT in `per_turn_results`

### Issue 2: Reward Tokens Not Extracted

**Problem**: Reward tokens are in `session_reward_audit[].raw_response.usage` but are never extracted and aggregated.

**Root Cause**:
- Reward audit entries contain `raw_response.usage` with token counts
- No code currently extracts and aggregates these tokens
- Need to iterate through `session_reward_audit` and sum up `raw_response.usage` values

**Evidence**:
- `ExecutionContext.set_session_reward(audit=...)` stores audit entries
- Each entry has `raw_response.usage` with token counts
- `_extract_atlas_metadata()` includes `session_reward_audit` but doesn't extract tokens from it

### Issue 3: ExecutionContext Lifecycle

**Problem**: ExecutionContext uses `ContextVar` which is async-context-local. If `atlas_arun()` runs in a different async context, tokens stored there won't be accessible.

**Root Cause**:
- `ContextVar` is scoped to async context
- If `atlas_arun()` creates a new async context or runs in a different context, ExecutionContext metadata won't be shared
- Need to verify ExecutionContext is accessible after `atlas_arun()` completes

**Evidence**:
- ExecutionContext uses `ContextVar` for thread-local/async-local storage
- `atlas_arun()` is async and may create nested contexts
- Tokens stored during `atlas_arun()` may not be accessible after it completes

## Recommended Solution

### Step 1: Extract Student Tokens from ExecutionContext

After `atlas_arun()` completes, check `ExecutionContext.metadata['token_usage']`:
- If it exists and has `prompt_tokens`/`completion_tokens`/`total_tokens` keys (flat structure)
- Move it to `ExecutionContext.metadata['token_usage']['student']`
- Preserve the `calls` count if needed

### Step 2: Extract Reward Tokens from session_reward_audit

After `atlas_arun()` completes, iterate through `ExecutionContext.metadata['session_reward_audit']`:
- For each entry, extract `raw_response.usage`
- Aggregate `prompt_tokens`, `completion_tokens`, `total_tokens`
- Store in `ExecutionContext.metadata['token_usage']['reward']`

### Step 3: Ensure ExecutionContext Access

Verify that ExecutionContext is accessible after `atlas_arun()` completes:
- Check if `ExecutionContext.get()` returns the same context instance
- If not, tokens may need to be extracted DURING `atlas_arun()` execution
- Consider using a callback or hook to extract tokens before context is lost

### Step 4: Update Extraction Function

Modify `_store_student_and_judge_token_usage()` to:
1. Extract student tokens from ExecutionContext (if flat structure exists)
2. Extract reward tokens from session_reward_audit
3. Extract judge tokens from per_turn_results (already implemented)
4. Store all in `ExecutionContext.metadata['token_usage']` with proper nesting

## Code Locations

### Atlas SDK Files
- `external/atlas-sdk/atlas/personas/student.py`: Student token tracking (lines 122-153)
- `external/atlas-sdk/atlas/personas/teacher.py`: Teacher/reward LLM calls (lines 90, 147)
- `external/atlas-sdk/atlas/learning/synthesizer.py`: Learning token tracking (lines 148-166)
- `external/atlas-sdk/atlas/runtime/orchestration/execution_context.py`: ExecutionContext implementation

### Integration Files
- `src/integration/atlas_integration.py`: Token extraction and metadata collection
- `src/integration/atlas_crm_adapter.py`: Conversation result compaction
- `src/evaluation/conversation_harness.py`: Judge token tracking

## Next Steps

1. **Verify ExecutionContext Access**: Test if ExecutionContext is accessible after `atlas_arun()` completes
2. **Implement Student Token Extraction**: Extract from ExecutionContext flat structure
3. **Implement Reward Token Extraction**: Aggregate from session_reward_audit
4. **Test Token Tracking**: Verify all tokens are captured and stored correctly
5. **Update Monitor Script**: Ensure monitor can detect tokens in new format

