import pytest

from src.pipeline.scenario_repository import ScenarioRepository


@pytest.fixture(scope="module")
def repo():
    return ScenarioRepository.from_default_paths()


def test_scenario_tags_capture_opportunity_stage(repo):
    tags = repo.scenario_tags["SC-00001"]
    assert tags["primary_entity"] == "Opportunity"
    assert tags["opportunity_stage"] in {"Negotiation", "Proposal", "Closed-Won", "Qualification"}
    assert tags["intent"] == "Opportunity Management"


def test_find_scenarios_filters_by_stage_and_success(repo):
    scenarios = repo.find_scenarios(
        expected_tool="modify_opportunity",
        expect_success=True,
        tag_filters={"opportunity_stage": {"Negotiation", "Proposal"}},
    )
    assert scenarios, "Expected at least one scenario matching stage filters"
    for record in scenarios:
        stage = repo.scenario_tags[record.scenario_id]["opportunity_stage"]
        assert stage in {"Negotiation", "Proposal"}
        assert record.expect_success is True
