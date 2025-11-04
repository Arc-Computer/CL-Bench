# Issue #26: Multi-Turn Conversation Architecture for CRM Benchmark

## Context

Customer requires multi-turn conversations (average 5-10 turns) to match production data patterns. Current system only supports single-turn scenarios (Issue #25). The benchmark needs to evaluate agents on realistic multi-turn workflows where users reference previous turns using natural language (pronouns, implicit references).

**Key Distinction**: 
- **Multi-tool** = 1 utterance → multiple tool calls (already supported)
- **Multi-turn** = N utterances → conversation with history (NEW requirement)

## Objectives

1. Implement conversation schema supporting 1-10 turns per conversation
2. Generate 1,500 conversations via Bespoke Curator using GPT-5-mini (900 simple, 450 medium, 150 complex)
3. Update harness to execute multi-turn conversations with history preservation
4. Maintain backward compatibility with existing single-turn scenarios

## Architecture Decisions

### Decision 1: Unified Conversation Architecture (Hybrid)
Support both single-turn and multi-turn as different complexity tiers:
- Simple (1-3 turns): Single-turn scenarios converted to 1-turn conversations
- Medium (4-6 turns): Client onboarding workflows
- Complex (7-10 turns): Full deal pipeline workflows

Rationale: Customer wants multi-turn, but simple tasks (create client, search opportunity) are still 1-2 turns in production. Allows graduated difficulty testing for Atlas learning.

### Decision 2: Explicit Template References (CRMArenaPro Pattern)
Use explicit `{{turn_N.field}}` syntax for cross-turn entity references in ground truth:

```python
Turn(
    turn_id=2,
    user_utterance="Create an opp for that client",  # Natural language
    expected_args={
        "client_id": "{{turn_1.client_id}}",  # Explicit template in ground truth
        "name": "Cloud Migration",
        "amount": 250000
    }
)
```

Rationale: Predictable and debuggable (no NLP ambiguity in ground truth). Agent still uses natural language; templates are for evaluation only.

### Decision 3: Complexity Distribution
1,500 conversations distributed as:

| Tier    | Turns | Count | %   | Example Workflows                                         |
|---------|-------|-------|-----|-----------------------------------------------------------|
| Simple  | 1-3   | 900   | 60% | Client search, create opportunity, modify quote           |
| Medium  | 4-6   | 450   | 30% | Client onboarding (create client → contact → opportunity) |
| Complex | 7-10  | 150   | 10% | Full deal pipeline (search → qualify → quote → contract)  |

Breakdown of Simple Tier:
- 400 single-turn (converted from existing scenario patterns)
- 300 two-turn (search → select)
- 200 three-turn (search → select → modify)

### Decision 4: Fail-Fast Execution
Stop conversation on first turn failure.

Rationale: Simpler evaluation logic (binary pass/fail), clear reward signals for Atlas, matches current scenario behavior.

## Implementation Phases

### Phase 1: Schema & Infrastructure
**Files to Create:**
- `src/conversation_schema.py` - Core schema (Turn, Conversation, Result)
- `src/conversation_templates.py` - 8 workflow templates
- `src/reference_resolver.py` - Resolves `{{turn_N.field}}` templates

**Files to Modify:**
- `src/scenario_generator.py` - Add `to_conversation()` method for backward compat

**Tests:**
- `tests/test_conversation_schema.py` - Schema validation
- `tests/test_reference_resolver.py` - Template resolution

**Deliverables:**
- Schema supports both single-turn and multi-turn
- Reference resolver validates templates before generation
- Backward compatibility: existing Scenario objects can convert to 1-turn Conversations

### Phase 2: Curator Multi-Turn Generator
**Files to Create:**
- `src/curator_conversation_generator.py` - Iterative multi-turn generation

**Generation Strategy:**
Iterative generation per conversation:
```python
for turn_idx in range(num_turns):
    prompt = build_turn_prompt(
        workflow_template=template,
        conversation_history=previous_turns,
        current_crm_state=entities,
        turn_number=turn_idx + 1
    )
    
    turn_response = curator_llm(prompt)  # GPT-5-mini batch mode
    
    # Update state for next turn
    entities = simulate_turn_execution(turn_response)
    previous_turns.append(turn_response)
```

**Prompts:**
- Turn 1: "Generate initial request for {workflow}"
- Turn 2+: "Given conversation history [...], generate next user utterance that naturally follows. Use pronouns like 'them', 'it' to reference previous entities."

**Deliverables:**
- Generate 1,500 conversations (900 simple, 450 medium, 150 complex)
- All conversations include realistic entity references
- Validation: All `{{turn_N.field}}` templates resolve correctly

### Phase 3: Multi-Turn Harness
**Files to Create:**
- `src/conversation_harness.py` - Multi-turn execution engine

**Core Logic:**
```python
def run_conversation(conv: Conversation) -> ConversationResult:
    backend = init_backend()
    setup_initial_entities(conv.initial_entities, backend)
    
    conversation_history = []
    for turn in conv.turns:
        # Resolve templates using previous turn results
        resolved_args = reference_resolver.resolve(
            turn.expected_args,
            previous_turns=conversation_history
        )
        
        # Build prompt with full history
        prompt = build_turn_prompt(turn, conversation_history, backend)
        
        # Get agent's tool call
        tool_call = agent.tool_call(turn, prompt)
        
        # Execute and validate
        result = execute_and_validate(tool_call, resolved_args, backend)
        
        if not result.success:
            # Fail-fast
            return ConversationResult(overall_success=False, failed_at_turn=turn.turn_id, ...)
        
        # Add to history
        conversation_history.append({
            "turn": turn.turn_id,
            "user": turn.user_utterance,
            "tool": tool_call.tool_name,
            "result": result
        })
    
    # All turns succeeded
    return ConversationResult(overall_success=True, ...)
```

**Files to Modify:**
- `src/run_baseline.py` - Add `--mode conversation` flag
- Update CLI to accept `conversations.jsonl` instead of `scenarios.jsonl`

**Deliverables:**
- Harness executes multi-turn conversations with history preservation
- Fail-fast on first turn failure
- Backward compatible: Can execute 1-turn conversations (converted scenarios)

### Phase 4: Validation & Testing
**Tests:**
- Smoke test: 10 conversations (3 simple, 5 medium, 2 complex)
- Validate reference resolution across turns
- Test fail-fast behavior
- Verify Atlas telemetry captures conversation-level metrics

**Deliverables:**
- All 1,500 conversations validated
- Mock agent: 100% success (ground truth)
- Integration tests pass

## Workflow Templates

Define 8 common CRM workflow patterns:

1. **Client Management** (simple, 1-3 turns): Create/search/modify client
2. **Contact Management** (simple, 2-3 turns): Search client → create contact
3. **Opportunity Management** (medium, 3-5 turns): Search → create → modify opportunity
4. **Quote Generation** (medium, 4-6 turns): Search opp → create quote → modify
5. **Client Onboarding** (medium, 5-6 turns): Create client → contact → opportunity
6. **Deal Pipeline** (complex, 7-10 turns): Search → qualify → quote → negotiate → contract
7. **Document Workflow** (simple, 2-3 turns): Search entity → upload document
8. **Multi-Entity Search** (medium, 4-5 turns): Search client → opportunities → quotes

## Dependencies

- **Blocks**: Issue #14 (Atlas integration), Issue #12 (baseline evaluations)
- **Supersedes**: Issue #25 (single-turn scenarios) - extends to multi-turn conversations
- **Uses**: Bespoke Curator with GPT-5-mini (already configured in `curator_dataset_generator.py`)

## Success Criteria

- [ ] 1,500 conversations generated with production-faithful patterns
- [ ] Harness executes multi-turn with conversation history
- [ ] All reference templates resolve correctly
- [ ] Mock agent achieves 100% success on conversations
- [ ] Backward compatible: Existing scenarios convert to 1-turn conversations
- [ ] Integration tests pass with Atlas telemetry capturing conversation metrics

## Backward Compatibility

### Scenario → Conversation Conversion
Existing Scenario objects automatically convert to 1-turn Conversation:
```python
def scenario_to_conversation(scenario: Scenario) -> Conversation:
    return Conversation(
        conversation_id=f"CONV-{scenario.scenario_id}",
        workflow_category=scenario.intent,
        complexity_level="simple",
        turns=[
            ConversationTurn(
                turn_id=1,
                user_utterance=scenario.utterance,
                expected_tool=scenario.expected_tool,
                expected_args=scenario.expected_args,
                expect_success=scenario.expect_success
            )
        ],
        initial_entities=scenario.setup_entities,
        success_criteria="all_turns"
    )
```

### Harness Compatibility
`conversation_harness.py` handles both:
- 1-turn conversations (backward compatible with scenarios)
- N-turn conversations (new multi-turn behavior)

Single code path, no branching needed.

## Risk Analysis

### Risk 1: Conversation Generation Quality
**Issue**: Multi-turn conversations may lack coherence or realistic context flow.

**Mitigation**: 
- Use explicit templates `{{turn_N.field}}` for entity tracking
- Validate all templates resolve before generation completes
- Human review of 50 sample conversations before full generation

### Risk 2: Reference Resolution Bugs
**Issue**: `{{turn_N.field}}` templates may reference non-existent turns or fields.

**Mitigation**:
- ReferenceResolver validates all templates at generation time
- Unit tests for edge cases (forward references, circular dependencies)
- Fail loudly during generation if templates invalid

### Risk 3: Atlas Integration Delay
**Issue**: Multi-turn changes may break Atlas SDK integration.

**Mitigation**:
- Conversation schema implements Gymnasium-compatible interface
- Atlas SDK autodiscovery should work unchanged
- Smoke test Atlas integration on 10 conversations before full run

## References

- [Bespoke Curator Documentation](https://docs.bespokelabs.ai/)
- Issue #25: Single-turn scenario generation (superseded by this issue)
- CRMArenaPro pattern for template references
- GPT-5-mini model configuration: `src/curator_dataset_generator.py:79`

