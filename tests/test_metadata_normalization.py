import copy

from analysis.normalize_seed_metadata import normalise_conversation


def test_normalise_conversation_applies_client_overrides():
    client_id = "075f99cd-17b5-4556-8b0f-1b68928dcfa6"
    conversation = {
        "conversation_id": "CHAIN-TEST-0001",
        "initial_entities": {
            "client_id": client_id,
            "client_name": "Legacy Name",
            "seed_data": {
                "Client": {
                    client_id: {
                        "client_id": client_id,
                        "name": "Legacy Name",
                        "status": "Prospect",
                    }
                }
            },
        },
        "turns": [
                {
                    "turn_id": 1,
                    "expected_tool": "modify_client",
                    "expected_args": {
                        "client_id": client_id,
                        "updates": {"status": "Active"},
                    },
                }
            ],
        "cumulative_context": {
            "turn_annotations": [
                {"turn_id": 1, "scenario_id": "SC-CLIENT-001"},
            ]
        },
    }
    overrides = {
            "SC-CLIENT-001": {
                "Client": {
                    "name": "TechVision Inc",
                    "status": "Active",
                    "email": "steve@techvision.com",
            }
        }
    }

    canonical_overrides = {}
    updated = normalise_conversation(copy.deepcopy(conversation), overrides, canonical_overrides=canonical_overrides)

    assert updated["initial_entities"]["client_name"] == "TechVision Inc"
    seed_record = updated["initial_entities"]["seed_data"]["Client"][client_id]
    assert seed_record["name"] == "TechVision Inc"
    assert seed_record["status"] == "Active"
    assert canonical_overrides["Client"][client_id]["name"] == "TechVision Inc"
