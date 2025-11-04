# Chained Workflows Documentation

This document describes the chained workflow generation system for multi-segment CRM conversations.

## Overview

Chained workflows enable generation of conversations that span multiple workflow segments, with proper entity state propagation and cumulative context tracking. This allows for realistic multi-turn conversations that mirror production CRM usage patterns.

## Architecture

### Workflow Chains

A `WorkflowChain` defines a sequence of workflow templates that form a multi-segment conversation:

```python
from src.conversation_templates import WorkflowChain, WORKFLOW_CHAINS

chain = WORKFLOW_CHAINS["onboarding_pipeline_contract"]
# Chain contains:
# - workflow_sequence: ["client_onboarding", "deal_pipeline", "quote_generation"]
# - success_pattern: [True, True, True]
# - entity_handoff_rules: {"client_id": "propagate", ...}
```

### Curator Structured Output

All Curator classes follow the Bespoke Curator patterns:

```python
from pydantic import BaseModel, Field
from bespokelabs import curator

class ResponseModel(BaseModel):
    items: List[ItemModel] = Field(description="...")

class CuratorLLM(curator.LLM):
    response_format = ResponseModel
    
    def prompt(self, input: Dict) -> str:
        return "..."  # Human-readable prompt
    
    def parse(self, input: Dict, response: ResponseModel) -> Dict:
        return [{"input_key": input["key"], **item.dict()} for item in response.items]
```

### Generation Process

1. **Scenario Selection**: `ScenarioSelector` selects scenarios for each turn
2. **Utterance Generation**: `ChainUtteranceGenerator` generates natural language utterances
3. **Entity Propagation**: Entity state carries across segments using `cumulative_context`
4. **Template Resolution**: Cross-turn references (`{{turn_N.field}}`) resolve correctly

## Usage

### Generating Chained Conversations

```python
from src.generation.chain_conversation_generator import instantiate_chained_conversation
from src.generation.chain_curator import ScenarioSelector, ChainUtteranceGenerator
from src.conversation_templates import WORKFLOW_CHAINS
from src.pipeline.scenario_repository import ScenarioRepository
import random

repo = ScenarioRepository.from_default_paths()
scenario_selector = ScenarioSelector(model_name="gpt-4.1-mini")
utterance_generator = ChainUtteranceGenerator(model_name="gpt-4.1-mini")
rng = random.Random(42)

chain = WORKFLOW_CHAINS["onboarding_pipeline_contract"]
conversation = instantiate_chained_conversation(
    chain,
    repo,
    scenario_selector,
    utterance_generator,
    rng,
    success_ratio=0.6,  # 60/40 success/failure split
)
```

### Executing Chained Conversations

```python
from src.evaluation.conversation_harness import ConversationHarness

harness = ConversationHarness([conversation])
results = harness.run()

result = results[0]
print(f"Chain success: {result.chain_success}")
print(f"Segments: {len(result.per_segment_results)}")
for segment in result.per_segment_results:
    print(f"  Segment {segment['segment_number']}: {segment['success']}")
```

## Template Resolution

Cross-turn references use `{{turn_N.field}}` syntax:

```python
{
    "client_id": "{{turn_1.client_id}}",  # References turn 1's client_id
    "amount": 50000
}
```

The `ReferenceResolver` validates and resolves these templates, ensuring:
- Turn numbers are sequential
- Referenced fields exist in previous turn results
- No circular dependencies

## Entity Handoff Rules

Chain segments transfer entities using handoff rules:

- `"propagate"`: Entity ID from previous segment is used
- `"create_in_segment_N"`: Entity is created in specified segment
- Custom rules can be defined per chain

## Reproduction Commands

### Generate Single Conversations

```bash
python scripts/generate_conversations.py \
    --count 100 \
    --seed 42 \
    --output-dir artifacts/conversations_multiturn
```

### Generate Chained Conversations

```bash
# TODO: Add chain generation script
python scripts/generate_chained_conversations.py \
    --chain onboarding_pipeline_contract \
    --count 50 \
    --seed 42
```

### Validate Conversations

```bash
python scripts/validate_chains.py \
    --conversations artifacts/conversations_multiturn/conversations.jsonl \
    --smoke-test
```

## Success/Failure Distribution

The system maintains a 60/40 success/failure distribution:
- **Scenario level**: 60% success, 40% failure scenarios
- **Turn level**: 60% success turns, 40% failure turns
- **Conversation level**: 60% success conversations, 40% failure conversations
- **Chain level**: 60% success chains, 40% failure chains

This is controlled via the `success_ratio` parameter (default 0.6).

## Validation

All generated conversations must:
1. Pass `ConversationHarness` validation
2. Resolve template references correctly
3. Propagate entity state across turns and segments
4. Comply with `fake_crm_tables_schema.json`
5. Execute successfully against mock harness

## No Fallbacks Policy

- Every turn must derive from validated scenarios
- Missing tools trigger generation, not fallbacks
- No placeholder values or artificial data
- All conversations must execute successfully

