"""Golden-case scenarios for CRM sandbox evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .crm_sandbox import Contract, MockCrmApi
from .validators import (
    CrmStateSnapshot,
    ValidationResult,
    validate_create_new_client,
    validate_create_new_opportunity,
    validate_create_quote,
    validate_modify_opportunity,
    validate_upload_document,
)

SetupFunc = Callable[[MockCrmApi], Dict[str, Any]]
BuildArgsFunc = Callable[[Dict[str, Any]], Dict[str, Any]]
BuildValidatorKwargsFunc = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
ValidatorFunc = Callable[[CrmStateSnapshot, CrmStateSnapshot, Mapping[str, Any]], ValidationResult]


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    task: str
    description: str
    utterance: str
    expected_tool: str
    setup: SetupFunc
    build_expected_args: BuildArgsFunc
    validator: ValidatorFunc
    build_validator_kwargs: Optional[BuildValidatorKwargsFunc] = None
    expect_success: bool = True
    expected_error_substring: Optional[str] = None
    tags: Sequence[str] = field(default_factory=tuple)

    def expected_args(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return dict(self.build_expected_args(context))

    def validator_kwargs(self, context: Dict[str, Any], expected_args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.build_validator_kwargs:
            return {}
        return self.build_validator_kwargs(context, dict(expected_args))


# ---------------------------------------------------------------------------
# Shared seeding helpers
# ---------------------------------------------------------------------------


def _seed_client(api: MockCrmApi, **client_kwargs: Any) -> Dict[str, Any]:
    client = api.create_new_client(**client_kwargs)
    return {"client": client}


def _ensure_opportunity(api: MockCrmApi, context: Dict[str, Any], **opp_kwargs: Any) -> Dict[str, Any]:
    opportunity = api.create_new_opportunity(client_id=context["client"].client_id, **opp_kwargs)
    updated = dict(context)
    updated["opportunity"] = opportunity
    return updated


def _ensure_quote(api: MockCrmApi, context: Dict[str, Any], **quote_kwargs: Any) -> Dict[str, Any]:
    quote = api.create_quote(opportunity_id=context["opportunity"].opportunity_id, **quote_kwargs)
    updated = dict(context)
    updated["quote"] = quote
    return updated


def _ensure_contract(api: MockCrmApi, context: Dict[str, Any], **contract_kwargs: Any) -> Dict[str, Any]:
    contract = Contract(
        client_id=context["client"].client_id,
        opportunity_id=context["opportunity"].opportunity_id,
        **contract_kwargs,
    )
    api.contracts[contract.contract_id] = contract
    updated = dict(context)
    updated["contract"] = contract
    return updated


# ---------------------------------------------------------------------------
# Create New Client cases
# ---------------------------------------------------------------------------


def _create_new_client_cases() -> List[GoldenCase]:
    raw_cases = [
        (
            "CNC-001",
            "Active SaaS client with phone",
            "Add Acme Analytics as an active client. Their main email is ops@acmeanalytics.example and phone is 415-555-0100.",
            {
                "name": "Acme Analytics",
                "email": "ops@acmeanalytics.example",
                "status": "Active",
                "phone": "415-555-0100",
            },
        ),
        (
            "CNC-002",
            "Prospect with industry",
            "Please register Borealis Labs as a prospect in the biotech industry using hello@borealis.example.",
            {
                "name": "Borealis Labs",
                "email": "hello@borealis.example",
                "status": "Prospect",
                "industry": "Biotech",
            },
        ),
        (
            "CNC-003",
            "Inactive client with address",
            "Log Delta Fabrication as inactive, located at 880 Industrial Blvd.",
            {
                "name": "Delta Fabrication",
                "email": "support@deltafab.example",
                "status": "Inactive",
                "address": "880 Industrial Blvd",
            },
        ),
        (
            "CNC-004",
            "Active client with owner",
            "Create a new active client called Ember Retail and assign it to owner stephanie.wong.",
            {
                "name": "Ember Retail",
                "email": "contact@emberretail.example",
                "status": "Active",
                "owner": "stephanie.wong",
            },
        ),
        (
            "CNC-005",
            "Prospect for today",
            "Set up Fresco Foods as a prospect added today with contact info team@frescofoods.example.",
            {
                "name": "Fresco Foods",
                "email": "team@frescofoods.example",
                "status": "Prospect",
            },
        ),
        (
            "CNC-006",
            "Active mobility account",
            "Please add Glider Mobility as an active client using mobility@glider.example.",
            {
                "name": "Glider Mobility",
                "email": "mobility@glider.example",
                "status": "Active",
            },
        ),
        (
            "CNC-007",
            "Inactive consulting firm",
            "Archive Harbor Consulting as inactive. Their phone is 212-555-0184.",
            {
                "name": "Harbor Consulting",
                "email": "info@harborconsulting.example",
                "status": "Inactive",
                "phone": "212-555-0184",
            },
        ),
        (
            "CNC-008",
            "Prospect owner assignment",
            "Create Indigo Health as a prospect and assign to owner federico.minutoli.",
            {
                "name": "Indigo Health",
                "email": "partnerships@indigohealth.example",
                "status": "Prospect",
                "owner": "federico.minutoli",
            },
        ),
        (
            "CNC-009",
            "Active logistics account",
            "Add Jetstream Logistics as an active transportation client at 44 Runway Rd.",
            {
                "name": "Jetstream Logistics",
                "email": "ops@jetstreamlogistics.example",
                "status": "Active",
                "address": "44 Runway Rd",
                "industry": "Transportation",
            },
        ),
        (
            "CNC-010",
            "Inactive supplier with owner",
            "Mark Keystone Supplies inactive, owner luca.carra, phone 646-555-0199.",
            {
                "name": "Keystone Supplies",
                "email": "sales@keystonesupplies.example",
                "status": "Inactive",
                "owner": "luca.carra",
                "phone": "646-555-0199",
            },
        ),
    ]

    cases: List[GoldenCase] = []
    for case_id, description, utterance, payload in raw_cases:
        def build_args(_: Dict[str, Any], payload=payload) -> Dict[str, Any]:
            return dict(payload)

        cases.append(
            GoldenCase(
                case_id=case_id,
                task="create_new_client",
                description=description,
                utterance=utterance,
                expected_tool="create_new_client",
                setup=lambda _: {},
                build_expected_args=build_args,
                validator=validate_create_new_client,
            )
        )
    return cases


def _create_new_client_negative_cases() -> List[GoldenCase]:
    cases: List[GoldenCase] = []

    cases.append(
        GoldenCase(
            case_id="CNC-101",
            task="create_new_client",
            description="Reject lower-case status value",
            utterance="Add Apex Holdings as an active client (status provided as lowercase).",
            expected_tool="create_new_client",
            setup=lambda _: {},
            build_expected_args=lambda _: {
                "name": "Apex Holdings",
                "email": "ops@apexholdings.example",
                "status": "active",
            },
            validator=validate_create_new_client,
            expect_success=False,
            expected_error_substring="Input should be",
        )
    )

    cases.append(
        GoldenCase(
            case_id="CNC-102",
            task="create_new_client",
            description="Reject malformed email",
            utterance="Create Beacon Labs with contact email beacon-labs-at-example.com.",
            expected_tool="create_new_client",
            setup=lambda _: {},
            build_expected_args=lambda _: {
                "name": "Beacon Labs",
                "email": "beacon-labs-at-example.com",
                "status": "Prospect",
            },
            validator=validate_create_new_client,
            expect_success=False,
            expected_error_substring=None,
        )
    )

    def duplicate_setup(api: MockCrmApi) -> Dict[str, Any]:
        return _seed_client(
            api,
            name="Catalyst Partners",
            email="hello@catalystpartners.example",
            status="Active",
        )

    cases.append(
        GoldenCase(
            case_id="CNC-103",
            task="create_new_client",
            description="Reject duplicate client email",
            utterance="Create Catalyst Partners again using hello@catalystpartners.example.",
            expected_tool="create_new_client",
            setup=duplicate_setup,
            build_expected_args=lambda _: {
                "name": "Catalyst Partners EU",
                "email": "hello@catalystpartners.example",
                "status": "Active",
            },
            validator=validate_create_new_client,
            expect_success=False,
            expected_error_substring="Client already exists with email",
        )
    )

    cases.append(
        GoldenCase(
            case_id="CNC-104",
            task="create_new_client",
            description="Reject missing email field",
            utterance="Add Helios Partners as a prospect; no email provided.",
            expected_tool="create_new_client",
            setup=lambda _: {},
            build_expected_args=lambda _: {
                "name": "Helios Partners",
                "status": "Prospect",
            },
            validator=validate_create_new_client,
            expect_success=False,
            expected_error_substring=None,
        )
    )

    return cases


# ---------------------------------------------------------------------------
# Create New Opportunity cases
# ---------------------------------------------------------------------------


def _create_new_opportunity_cases() -> List[GoldenCase]:
    raw_cases = [
        (
            "CNO-001",
            "Net-new opportunity in Prospecting",
            "Log a new opportunity called Acme Renewal for $85K at the Prospecting stage.",
            {"name": "Acme Analytics", "email": "ops@acmeanalytics.example", "status": "Active"},
            {"name": "Acme Renewal", "amount": 85_000.0, "stage": "Prospecting"},
        ),
        (
            "CNO-002",
            "Qualification stage deal with owner",
            "Create an opportunity named Borealis Pilot for Borealis Labs at $120K in Qualification, owner stephanie.wong.",
            {"name": "Borealis Labs", "email": "hello@borealis.example", "status": "Prospect"},
            {"name": "Borealis Pilot", "amount": 120_000.0, "stage": "Qualification", "owner": "stephanie.wong", "probability": 25},
        ),
        (
            "CNO-003",
            "Proposal stage with notes",
            "Open a Proposal stage deal called Delta Fabrication Upgrade at $210K.",
            {"name": "Delta Fabrication", "email": "support@deltafab.example", "status": "Active"},
            {"name": "Delta Fabrication Upgrade", "amount": 210_000.0, "stage": "Proposal", "notes": "Upgrade package requested"},
        ),
        (
            "CNO-004",
            "Negotiation stage with close date",
            "Track Ember Retail Expansion at $330K in Negotiation scheduled to close on 2025-11-30.",
            {"name": "Ember Retail", "email": "contact@emberretail.example", "status": "Active"},
            {"name": "Ember Retail Expansion", "amount": 330_000.0, "stage": "Negotiation", "close_date": "2025-11-30"},
        ),
        (
            "CNO-005",
            "Closed-Won migration",
            "Record Fresco Foods Renewal closed-won at $95K.",
            {"name": "Fresco Foods", "email": "team@frescofoods.example", "status": "Active"},
            {"name": "Fresco Foods Renewal", "amount": 95_000.0, "stage": "Closed-Won"},
        ),
        (
            "CNO-006",
            "Closed-Lost documentation",
            "Add Glider Mobility Pilot as Closed-Lost for $140K and note competition from Hawk Systems.",
            {"name": "Glider Mobility", "email": "mobility@glider.example", "status": "Prospect"},
            {"name": "Glider Mobility Pilot", "amount": 140_000.0, "stage": "Closed-Lost", "notes": "Lost to Hawk Systems due to pricing"},
        ),
        (
            "CNO-007",
            "Negotiation with probability",
            "Enter Harbor Consulting Expansion at $175K in Negotiation with probability 55%.",
            {"name": "Harbor Consulting", "email": "info@harborconsulting.example", "status": "Active"},
            {"name": "Harbor Consulting Expansion", "amount": 175_000.0, "stage": "Negotiation", "probability": 55},
        ),
        (
            "CNO-008",
            "Qualification with owner notes",
            "Create Indigo Health Pilot at $210K in Qualification, assign to aman.sharma.",
            {"name": "Indigo Health", "email": "partnerships@indigohealth.example", "status": "Prospect"},
            {"name": "Indigo Health Pilot", "amount": 210_000.0, "stage": "Qualification", "owner": "aman.sharma"},
        ),
        (
            "CNO-009",
            "Prospecting with close date and probability",
            "Start Jetstream Logistics Modernization at $420K, Prospecting, probability 20%, close date 2025-12-15.",
            {"name": "Jetstream Logistics", "email": "ops@jetstreamlogistics.example", "status": "Active"},
            {"name": "Jetstream Logistics Modernization", "amount": 420_000.0, "stage": "Prospecting", "probability": 20, "close_date": "2025-12-15"},
        ),
        (
            "CNO-010",
            "Negotiation with owner and notes",
            "Log Keystone Supplies Expansion at $260K in Negotiation assigned to gaby.chan with note 'requires custom SLA'.",
            {"name": "Keystone Supplies", "email": "sales@keystonesupplies.example", "status": "Active"},
            {"name": "Keystone Supplies Expansion", "amount": 260_000.0, "stage": "Negotiation", "owner": "gaby.chan", "notes": "Requires custom SLA"},
        ),
    ]

    cases: List[GoldenCase] = []
    for case_id, description, utterance, client_payload, opportunity_payload in raw_cases:
        def setup(api: MockCrmApi, payload=client_payload) -> Dict[str, Any]:
            return _seed_client(api, **payload)

        def build_args(context: Dict[str, Any], opp_payload=opportunity_payload) -> Dict[str, Any]:
            args: Dict[str, Any] = {
                "client_id": context["client"].client_id,
                "name": opp_payload["name"],
                "amount": opp_payload["amount"],
                "stage": opp_payload["stage"],
            }
            for optional_key in ("owner", "probability", "notes", "close_date"):
                if optional_key in opp_payload:
                    args[optional_key] = opp_payload[optional_key]
            return args

        cases.append(
            GoldenCase(
                case_id=case_id,
                task="create_new_opportunity",
                description=description,
                utterance=utterance,
                expected_tool="create_new_opportunity",
                setup=setup,
                build_expected_args=build_args,
                validator=validate_create_new_opportunity,
            )
        )
    return cases


def _create_new_opportunity_negative_cases() -> List[GoldenCase]:
    cases: List[GoldenCase] = []

    def base_client(api: MockCrmApi, name: str, email: str, status: str = "Active") -> Dict[str, Any]:
        return _seed_client(api, name=name, email=email, status=status)

    cases.append(
        GoldenCase(
            case_id="CNO-101",
            task="create_new_opportunity",
            description="Reject invalid stage value",
            utterance="Create Nimbus Analytics Expansion at Negotiations stage (typo).",
            expected_tool="create_new_opportunity",
            setup=lambda api: base_client(api, "Nimbus Analytics", "info@nimbus.example"),
            build_expected_args=lambda context: {
                "client_id": context["client"].client_id,
                "name": "Nimbus Analytics Expansion",
                "amount": 150_000.0,
                "stage": "Negotiations",
            },
            validator=validate_create_new_opportunity,
            expect_success=False,
            expected_error_substring="Input should be",
        )
    )

    cases.append(
        GoldenCase(
            case_id="CNO-102",
            task="create_new_opportunity",
            description="Reject probability above 100%",
            utterance="Log Orion Systems Pilot with probability 125%.",
            expected_tool="create_new_opportunity",
            setup=lambda api: base_client(api, "Orion Systems", "sales@orionsystems.example", status="Prospect"),
            build_expected_args=lambda context: {
                "client_id": context["client"].client_id,
                "name": "Orion Systems Pilot",
                "amount": 200_000.0,
                "stage": "Qualification",
                "probability": 125,
            },
            validator=validate_create_new_opportunity,
            expect_success=False,
            expected_error_substring="less than or equal to 100",
        )
    )

    cases.append(
        GoldenCase(
            case_id="CNO-103",
            task="create_new_opportunity",
            description="Reject opportunity referencing unknown client",
            utterance="Create Phoenix Group Expansion linked to client 0000-unknown-id.",
            expected_tool="create_new_opportunity",
            setup=lambda _: {},
            build_expected_args=lambda _: {
                "client_id": "00000000-0000-0000-0000-unknown",
                "name": "Phoenix Group Expansion",
                "amount": 180_000.0,
                "stage": "Prospecting",
            },
            validator=validate_create_new_opportunity,
            expect_success=False,
            expected_error_substring="Client not found with ID",
        )
    )

    cases.append(
        GoldenCase(
            case_id="CNO-104",
            task="create_new_opportunity",
            description="Reject invalid close_date format",
            utterance="Set Vega Aerospace Upgrade to close on 12/15/2025 (US format).",
            expected_tool="create_new_opportunity",
            setup=lambda api: base_client(api, "Vega Aerospace", "team@vegaaero.example"),
            build_expected_args=lambda context: {
                "client_id": context["client"].client_id,
                "name": "Vega Aerospace Upgrade",
                "amount": 240_000.0,
                "stage": "Negotiation",
                "close_date": "12/15/2025",
            },
            validator=validate_create_new_opportunity,
            expect_success=False,
            expected_error_substring="valid date",
        )
    )

    cases.append(
        GoldenCase(
            case_id="CNO-105",
            task="create_new_opportunity",
            description="Reject probability expressed as text",
            utterance="Create Zenith Capital Pilot and set probability to 'fifty'.",
            expected_tool="create_new_opportunity",
            setup=lambda api: base_client(api, "Zenith Capital", "hello@zenithcapital.example", status="Prospect"),
            build_expected_args=lambda context: {
                "client_id": context["client"].client_id,
                "name": "Zenith Capital Pilot",
                "amount": 175_000.0,
                "stage": "Qualification",
                "probability": "fifty",
            },
            validator=validate_create_new_opportunity,
            expect_success=False,
            expected_error_substring="Input should be",
        )
    )

    return cases
def _create_quote_cases() -> List[GoldenCase]:
    raw_cases = [
        (
            "CQT-001",
            "Draft quote for renewal",
            "Generate a draft quote for Acme Renewal worth $85K with version v1.",
            {"name": "Acme Analytics", "email": "ops@acmeanalytics.example", "status": "Active"},
            {"name": "Acme Renewal", "amount": 85_000.0, "stage": "Prospecting"},
            {"amount": 85_000.0, "status": "Draft", "version": "v1"},
        ),
        (
            "CQT-002",
            "Sent quote with prefix",
            "Send the Borealis Pilot quote for $120K, mark status Sent and prefix BP-2025.",
            {"name": "Borealis Labs", "email": "hello@borealis.example", "status": "Prospect"},
            {"name": "Borealis Pilot", "amount": 120_000.0, "stage": "Qualification"},
            {"amount": 120_000.0, "status": "Sent", "quote_prefix": "BP-2025"},
        ),
        (
            "CQT-003",
            "Approved quote with expiry",
            "Issue an approved quote for Delta Fabrication Upgrade at $210K valid until 2025-11-30.",
            {"name": "Delta Fabrication", "email": "support@deltafab.example", "status": "Active"},
            {"name": "Delta Fabrication Upgrade", "amount": 210_000.0, "stage": "Proposal"},
            {"amount": 210_000.0, "status": "Approved", "valid_until": "2025-11-30"},
        ),
        (
            "CQT-004",
            "Rejected quote",
            "Create a rejected quote for Ember Retail Expansion at $330K.",
            {"name": "Ember Retail", "email": "contact@emberretail.example", "status": "Active"},
            {"name": "Ember Retail Expansion", "amount": 330_000.0, "stage": "Negotiation"},
            {"amount": 330_000.0, "status": "Rejected"},
        ),
        (
            "CQT-005",
            "Canceled quote",
            "Cancel the Fresco Foods Renewal quote for $95K.",
            {"name": "Fresco Foods", "email": "team@frescofoods.example", "status": "Active"},
            {"name": "Fresco Foods Renewal", "amount": 95_000.0, "stage": "Closed-Won"},
            {"amount": 95_000.0, "status": "Canceled"},
        ),
        (
            "CQT-006",
            "Second version draft",
            "Prepare version v2 draft quote for Glider Mobility Pilot at $140K.",
            {"name": "Glider Mobility", "email": "mobility@glider.example", "status": "Prospect"},
            {"name": "Glider Mobility Pilot", "amount": 140_000.0, "stage": "Prospecting"},
            {"amount": 140_000.0, "status": "Draft", "version": "v2"},
        ),
        (
            "CQT-007",
            "Sent quote with prefix and version",
            "Send Harbor Consulting Expansion quote version 1.1 for $175K with prefix HC-EXP.",
            {"name": "Harbor Consulting", "email": "info@harborconsulting.example", "status": "Active"},
            {"name": "Harbor Consulting Expansion", "amount": 175_000.0, "stage": "Negotiation"},
            {"amount": 175_000.0, "status": "Sent", "version": "1.1", "quote_prefix": "HC-EXP"},
        ),
        (
            "CQT-008",
            "Approved quote with future expiry",
            "Approve Indigo Health Pilot quote for $210K, valid until 2025-12-31.",
            {"name": "Indigo Health", "email": "partnerships@indigohealth.example", "status": "Prospect"},
            {"name": "Indigo Health Pilot", "amount": 210_000.0, "stage": "Qualification"},
            {"amount": 210_000.0, "status": "Approved", "valid_until": "2025-12-31"},
        ),
        (
            "CQT-009",
            "Draft quote with prefix",
            "Create Jetstream Logistics Modernization draft quote for $420K with prefix JET-2025.",
            {"name": "Jetstream Logistics", "email": "ops@jetstreamlogistics.example", "status": "Active"},
            {"name": "Jetstream Logistics Modernization", "amount": 420_000.0, "stage": "Prospecting"},
            {"amount": 420_000.0, "status": "Draft", "quote_prefix": "JET-2025"},
        ),
        (
            "CQT-010",
            "Approved quote with version info",
            "Approve Keystone Supplies Expansion quote version V3 at $260K.",
            {"name": "Keystone Supplies", "email": "sales@keystonesupplies.example", "status": "Active"},
            {"name": "Keystone Supplies Expansion", "amount": 260_000.0, "stage": "Negotiation"},
            {"amount": 260_000.0, "status": "Approved", "version": "V3"},
        ),
    ]

    cases: List[GoldenCase] = []
    for case_id, description, utterance, client_payload, opportunity_payload, quote_payload in raw_cases:
        def setup(api: MockCrmApi, client_data=client_payload, opp_data=opportunity_payload) -> Dict[str, Any]:
            context = _seed_client(api, **client_data)
            return _ensure_opportunity(api, context, **opp_data)

        def build_args(context: Dict[str, Any], payload=quote_payload) -> Dict[str, Any]:
            args: Dict[str, Any] = {
                "opportunity_id": context["opportunity"].opportunity_id,
                "amount": payload["amount"],
                "status": payload["status"],
            }
            for optional_key in ("version", "valid_until", "quote_prefix"):
                if optional_key in payload:
                    args[optional_key] = payload[optional_key]
            return args

        cases.append(
            GoldenCase(
                case_id=case_id,
                task="create_quote",
                description=description,
                utterance=utterance,
                expected_tool="create_quote",
                setup=setup,
                build_expected_args=build_args,
                validator=validate_create_quote,
            )
        )
    return cases


def _create_quote_negative_cases() -> List[GoldenCase]:
    cases: List[GoldenCase] = []

    def base_opportunity(api: MockCrmApi, client_name: str, client_email: str, opp_name: str, amount: float, stage: str = "Proposal") -> Dict[str, Any]:
        context = _seed_client(api, name=client_name, email=client_email, status="Active")
        return _ensure_opportunity(api, context, name=opp_name, amount=amount, stage=stage)

    cases.append(
        GoldenCase(
            case_id="CQT-101",
            task="create_quote",
            description="Reject lowercase quote status",
            utterance="Send the Lumina Ventures Renewal quote and mark status approved (lowercase).",
            expected_tool="create_quote",
            setup=lambda api: base_opportunity(api, "Lumina Ventures", "contact@lumina.example", "Lumina Ventures Renewal", 95_000.0),
            build_expected_args=lambda context: {
                "opportunity_id": context["opportunity"].opportunity_id,
                "amount": 95_000.0,
                "status": "approved",
            },
            validator=validate_create_quote,
            expect_success=False,
            expected_error_substring="Input should be",
        )
    )

    cases.append(
        GoldenCase(
            case_id="CQT-102",
            task="create_quote",
            description="Reject negative quote amount",
            utterance="Create a quote for Mosaic Retail Expansion with amount -5000 (credit note).",
            expected_tool="create_quote",
            setup=lambda api: base_opportunity(api, "Mosaic Retail", "hello@mosaicretail.example", "Mosaic Retail Expansion", 150_000.0, stage="Negotiation"),
            build_expected_args=lambda context: {
                "opportunity_id": context["opportunity"].opportunity_id,
                "amount": -5_000.0,
                "status": "Draft",
            },
            validator=validate_create_quote,
            expect_success=False,
            expected_error_substring="greater than or equal to 0",
        )
    )

    cases.append(
        GoldenCase(
            case_id="CQT-103",
            task="create_quote",
            description="Reject quote with unknown opportunity",
            utterance="Issue a quote against opportunity 0000-unknown-id for $50K.",
            expected_tool="create_quote",
            setup=lambda _: {},
            build_expected_args=lambda _: {
                "opportunity_id": "00000000-0000-0000-0000-unknown",
                "amount": 50_000.0,
                "status": "Draft",
            },
            validator=validate_create_quote,
            expect_success=False,
            expected_error_substring="Opportunity not found with ID",
        )
    )

    cases.append(
        GoldenCase(
            case_id="CQT-104",
            task="create_quote",
            description="Reject invalid quote status value",
            utterance="Submit a quote for Horizon Labs and mark status InReview.",
            expected_tool="create_quote",
            setup=lambda api: base_opportunity(api, "Horizon Labs", "team@horizonlabs.example", "Horizon Labs Upgrade", 130_000.0),
            build_expected_args=lambda context: {
                "opportunity_id": context["opportunity"].opportunity_id,
                "amount": 130_000.0,
                "status": "InReview",
            },
            validator=validate_create_quote,
            expect_success=False,
            expected_error_substring="Input should be",
        )
    )

    cases.append(
        GoldenCase(
            case_id="CQT-105",
            task="create_quote",
            description="Reject missing amount field",
            utterance="Create a draft quote for Ionis Manufacturing but omit the amount.",
            expected_tool="create_quote",
            setup=lambda api: base_opportunity(api, "Ionis Manufacturing", "sales@ionis.example", "Ionis Manufacturing Renewal", 210_000.0),
            build_expected_args=lambda context: {
                "opportunity_id": context["opportunity"].opportunity_id,
                "status": "Draft",
            },
            validator=validate_create_quote,
            expect_success=False,
            expected_error_substring=None,
        )
    )

    return cases
def _upload_document_cases() -> List[GoldenCase]:
    cases: List[GoldenCase] = []

    def make_case(
        case_id: str,
        description: str,
        utterance: str,
        entity_type: str,
        file_name: str,
        setup: SetupFunc,
    ) -> GoldenCase:
        def build_args(context: Dict[str, Any]) -> Dict[str, Any]:
            if entity_type == "Client":
                entity_id = context["client"].client_id
            elif entity_type == "Opportunity":
                entity_id = context["opportunity"].opportunity_id
            elif entity_type == "Quote":
                entity_id = context["quote"].quote_id
            elif entity_type == "Contract":
                entity_id = context["contract"].contract_id
            else:
                raise ValueError(f"Unsupported entity type '{entity_type}'.")
            return {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "file_name": file_name,
            }

        return GoldenCase(
            case_id=case_id,
            task="upload_document",
            description=description,
            utterance=utterance,
            expected_tool="upload_document",
            setup=setup,
            build_expected_args=build_args,
            validator=validate_upload_document,
        )

    def setup_client(api: MockCrmApi) -> Dict[str, Any]:
        return _seed_client(api, name="Acme Analytics", email="ops@acmeanalytics.example", status="Active")

    def setup_opportunity(api: MockCrmApi) -> Dict[str, Any]:
        context = _seed_client(api, name="Borealis Labs", email="hello@borealis.example", status="Prospect")
        return _ensure_opportunity(api, context, name="Borealis Pilot", amount=120_000.0, stage="Qualification")

    def setup_quote(api: MockCrmApi) -> Dict[str, Any]:
        context = _seed_client(api, name="Delta Fabrication", email="support@deltafab.example", status="Active")
        context = _ensure_opportunity(api, context, name="Delta Fabrication Upgrade", amount=210_000.0, stage="Proposal")
        return _ensure_quote(api, context, amount=210_000.0, status="Draft")

    def setup_contract(api: MockCrmApi) -> Dict[str, Any]:
        context = _seed_client(api, name="Ember Retail", email="contact@emberretail.example", status="Active")
        context = _ensure_opportunity(api, context, name="Ember Retail Expansion", amount=330_000.0, stage="Negotiation")
        return _ensure_contract(api, context, status="Active", value=330_000.0)

    uploads = [
        ("UD-001", "Client onboarding brief", "Upload the client onboarding brief for Acme Analytics (acme-onboarding.pdf).", "Client", "acme-onboarding.pdf", setup_client),
        ("UD-002", "Client compliance memo", "Attach the compliance memo to Acme Analytics titled acme-compliance-memo.docx.", "Client", "acme-compliance-memo.docx", setup_client),
        ("UD-003", "Client kickoff deck", "Upload Acme Analytics kickoff deck titled acme-kickoff.key.", "Client", "acme-kickoff.key", setup_client),
        ("UD-004", "Opportunity proposal deck", "Attach the proposal deck to the Borealis Pilot opportunity as Borealis-Pilot-Deck.pptx.", "Opportunity", "Borealis-Pilot-Deck.pptx", setup_opportunity),
        ("UD-005", "Opportunity ROI analysis", "Link the ROI analysis spreadsheet to Borealis Pilot called borealis-roi.xlsx.", "Opportunity", "borealis-roi.xlsx", setup_opportunity),
        ("UD-006", "Opportunity statement of work", "Attach the statement of work to Borealis Pilot as borealis-sow.pdf.", "Opportunity", "borealis-sow.pdf", setup_opportunity),
        ("UD-007", "Quote PDF", "Upload the signed quote PDF for Delta Fabrication Upgrade called delta-upgrade-quote.pdf.", "Quote", "delta-upgrade-quote.pdf", setup_quote),
        ("UD-008", "Quote redline", "Upload the redlined quote for Delta Fabrication Upgrade named delta-quote-redline.docx.", "Quote", "delta-quote-redline.docx", setup_quote),
        ("UD-009", "Contract document", "Add the executed contract for Ember Retail Expansion named ember-expansion-contract.pdf.", "Contract", "ember-expansion-contract.pdf", setup_contract),
        ("UD-010", "Contract amendment", "Attach the amendment for Ember Retail Expansion contract as ember-amendment.pdf.", "Contract", "ember-amendment.pdf", setup_contract),
    ]

    for data in uploads:
        cases.append(make_case(*data))
    return cases


def _upload_document_negative_cases() -> List[GoldenCase]:
    cases: List[GoldenCase] = []

    def client_only(api: MockCrmApi) -> Dict[str, Any]:
        return _seed_client(api, name="Nova Manufacturing", email="team@novamfg.example", status="Active")

    cases.append(
        GoldenCase(
            case_id="UD-101",
            task="upload_document",
            description="Reject lowercase entity_type",
            utterance="Upload nova-playbook.pdf to client Nova Manufacturing (entity_type lowercase).",
            expected_tool="upload_document",
            setup=client_only,
            build_expected_args=lambda context: {
                "entity_type": "client",
                "entity_id": context["client"].client_id,
                "file_name": "nova-playbook.pdf",
            },
            validator=validate_upload_document,
            expect_success=False,
            expected_error_substring="valid DocumentEntityType",
        )
    )

    cases.append(
        GoldenCase(
            case_id="UD-102",
            task="upload_document",
            description="Reject document referencing missing opportunity",
            utterance="Attach lost-deck.pdf to opportunity 0000-unknown-id.",
            expected_tool="upload_document",
            setup=lambda _: {},
            build_expected_args=lambda _: {
                "entity_type": "Opportunity",
                "entity_id": "00000000-0000-0000-0000-unknown",
                "file_name": "lost-deck.pdf",
            },
            validator=validate_upload_document,
            expect_success=False,
            expected_error_substring=None,
        )
    )

    cases.append(
        GoldenCase(
            case_id="UD-103",
            task="upload_document",
            description="Reject non-UUID entity_id",
            utterance="Attach compliance-note.pdf to client using an invalid ID string.",
            expected_tool="upload_document",
            setup=client_only,
            build_expected_args=lambda _: {
                "entity_type": "Client",
                "entity_id": "not-a-uuid",
                "file_name": "compliance-note.pdf",
            },
            validator=validate_upload_document,
            expect_success=False,
            expected_error_substring=None,
        )
    )

    return cases
def _modify_opportunity_cases() -> List[GoldenCase]:
    raw_cases = [
        (
            "MOP-001",
            "Stage progression with probability",
            "Update Acme Renewal to move from Prospecting to Qualification with probability 35%.",
            {"name": "Acme Analytics", "email": "ops@acmeanalytics.example", "status": "Active"},
            {"name": "Acme Renewal", "amount": 85_000.0, "stage": "Prospecting"},
            {"stage": "Qualification", "probability": 35},
        ),
        (
            "MOP-002",
            "Negotiation amount increase",
            "Increase Borealis Pilot amount to $135K and mark stage as Negotiation.",
            {"name": "Borealis Labs", "email": "hello@borealis.example", "status": "Prospect"},
            {"name": "Borealis Pilot", "amount": 120_000.0, "stage": "Qualification"},
            {"amount": 135_000.0, "stage": "Negotiation"},
        ),
        (
            "MOP-003",
            "Add notes on competitor",
            "Add a note to Delta Fabrication Upgrade that EuroFab is discounting heavily and set probability to 30%.",
            {"name": "Delta Fabrication", "email": "support@deltafab.example", "status": "Active"},
            {"name": "Delta Fabrication Upgrade", "amount": 210_000.0, "stage": "Proposal"},
            {"notes": "EuroFab discounting aggressively", "probability": 30},
        ),
        (
            "MOP-004",
            "Adjust close date",
            "Push Ember Retail Expansion close date to 2025-12-15 and set probability 70%.",
            {"name": "Ember Retail", "email": "contact@emberretail.example", "status": "Active"},
            {"name": "Ember Retail Expansion", "amount": 330_000.0, "stage": "Negotiation"},
            {"close_date": "2025-12-15", "probability": 70},
        ),
        (
            "MOP-005",
            "Mark as Closed-Won",
            "Mark Fresco Foods Renewal as Closed-Won and note that procurement approved the terms.",
            {"name": "Fresco Foods", "email": "team@frescofoods.example", "status": "Active"},
            {"name": "Fresco Foods Renewal", "amount": 95_000.0, "stage": "Negotiation"},
            {"stage": "Closed-Won", "notes": "Procurement approved final terms"},
        ),
        (
            "MOP-006",
            "Stage regression to Proposal",
            "Move Glider Mobility Pilot back to Proposal and reduce amount to $125K.",
            {"name": "Glider Mobility", "email": "mobility@glider.example", "status": "Prospect"},
            {"name": "Glider Mobility Pilot", "amount": 140_000.0, "stage": "Negotiation"},
            {"stage": "Proposal", "amount": 125_000.0},
        ),
        (
            "MOP-007",
            "Update owner and probability",
            "Assign Harbor Consulting Expansion to aman.sharma and set probability 65%.",
            {"name": "Harbor Consulting", "email": "info@harborconsulting.example", "status": "Active"},
            {"name": "Harbor Consulting Expansion", "amount": 175_000.0, "stage": "Negotiation"},
            {"owner": "aman.sharma", "probability": 65},
        ),
        (
            "MOP-008",
            "Document escalation note",
            "Add note to Indigo Health Pilot that legal needs to review terms.",
            {"name": "Indigo Health", "email": "partnerships@indigohealth.example", "status": "Prospect"},
            {"name": "Indigo Health Pilot", "amount": 210_000.0, "stage": "Qualification"},
            {"notes": "Legal review required before next call"},
        ),
        (
            "MOP-009",
            "Adjust probability downward",
            "Reduce Jetstream Logistics Modernization probability to 15% and update notes accordingly.",
            {"name": "Jetstream Logistics", "email": "ops@jetstreamlogistics.example", "status": "Active"},
            {"name": "Jetstream Logistics Modernization", "amount": 420_000.0, "stage": "Prospecting"},
            {"probability": 15, "notes": "Customer prioritizing ERP rollout first"},
        ),
        (
            "MOP-010",
            "Close lost with reason",
            "Set Keystone Supplies Expansion to Closed-Lost and note competitor NovaEdge.",
            {"name": "Keystone Supplies", "email": "sales@keystonesupplies.example", "status": "Active"},
            {"name": "Keystone Supplies Expansion", "amount": 260_000.0, "stage": "Negotiation"},
            {"stage": "Closed-Lost", "notes": "Lost to NovaEdge due to pricing"},
        ),
    ]

    cases: List[GoldenCase] = []
    for case_id, description, utterance, client_payload, opportunity_payload, updates in raw_cases:
        def setup(api: MockCrmApi, client_data=client_payload, opp_data=opportunity_payload) -> Dict[str, Any]:
            context = _seed_client(api, **client_data)
            return _ensure_opportunity(api, context, **opp_data)

        def build_args(context: Dict[str, Any], updates_payload=updates) -> Dict[str, Any]:
            return {
                "opportunity_id": context["opportunity"].opportunity_id,
                "updates": dict(updates_payload),
            }

        def build_kwargs(_: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
            return {"updates": expected["updates"]}

        cases.append(
            GoldenCase(
                case_id=case_id,
                task="modify_opportunity",
                description=description,
                utterance=utterance,
                expected_tool="modify_opportunity",
                setup=setup,
                build_expected_args=build_args,
                validator=validate_modify_opportunity,
                build_validator_kwargs=build_kwargs,
            )
        )
    return cases


def _modify_opportunity_negative_cases() -> List[GoldenCase]:
    cases: List[GoldenCase] = []

    def base_opportunity(api: MockCrmApi, client_name: str, email: str, opp_name: str, amount: float, stage: str) -> Dict[str, Any]:
        context = _seed_client(api, name=client_name, email=email, status="Active")
        return _ensure_opportunity(api, context, name=opp_name, amount=amount, stage=stage)

    cases.append(
        GoldenCase(
            case_id="MOP-101",
            task="modify_opportunity",
            description="Reject invalid stage transition value",
            utterance="Set Orbit Media Upgrade stage to Negotiations (typo).",
            expected_tool="modify_opportunity",
            setup=lambda api: base_opportunity(api, "Orbit Media", "contact@orbitmedia.example", "Orbit Media Upgrade", 140_000.0, "Negotiation"),
            build_expected_args=lambda context: {
                "opportunity_id": context["opportunity"].opportunity_id,
                "updates": {"stage": "Negotiations"},
            },
            validator=validate_modify_opportunity,
            build_validator_kwargs=lambda _, expected: {"updates": expected["updates"]},
            expect_success=False,
            expected_error_substring="Input should be",
        )
    )

    cases.append(
        GoldenCase(
            case_id="MOP-102",
            task="modify_opportunity",
            description="Reject probability below zero",
            utterance="Update Parallax Finance Pilot probability to -10%.",
            expected_tool="modify_opportunity",
            setup=lambda api: base_opportunity(api, "Parallax Finance", "hello@parallax.example", "Parallax Finance Pilot", 220_000.0, "Qualification"),
            build_expected_args=lambda context: {
                "opportunity_id": context["opportunity"].opportunity_id,
                "updates": {"probability": -10},
            },
            validator=validate_modify_opportunity,
            build_validator_kwargs=lambda _, expected: {"updates": expected["updates"]},
            expect_success=False,
            expected_error_substring="greater than or equal to 0",
        )
    )

    cases.append(
        GoldenCase(
            case_id="MOP-103",
            task="modify_opportunity",
            description="Reject updates to unsupported field",
            utterance="Assign Quasar Logistics opportunity to stephanie.wong via assigned_to field.",
            expected_tool="modify_opportunity",
            setup=lambda api: base_opportunity(api, "Quasar Logistics", "ops@quasarlogistics.example", "Quasar Logistics Route Optimization", 310_000.0, "Negotiation"),
            build_expected_args=lambda context: {
                "opportunity_id": context["opportunity"].opportunity_id,
                "updates": {"assigned_to": "stephanie.wong"},
            },
            validator=validate_modify_opportunity,
            build_validator_kwargs=lambda _, expected: {"updates": expected["updates"]},
            expect_success=False,
            expected_error_substring="Opportunity has no field named 'assigned_to'",
        )
    )

    cases.append(
        GoldenCase(
            case_id="MOP-104",
            task="modify_opportunity",
            description="Reject probability expressed as text",
            utterance="Set Vega Analytics Pilot probability to 'high'.",
            expected_tool="modify_opportunity",
            setup=lambda api: base_opportunity(api, "Vega Analytics", "team@vegaanalytics.example", "Vega Analytics Pilot", 190_000.0, "Qualification"),
            build_expected_args=lambda context: {
                "opportunity_id": context["opportunity"].opportunity_id,
                "updates": {"probability": "high"},
            },
            validator=validate_modify_opportunity,
            build_validator_kwargs=lambda _, expected: {"updates": expected["updates"]},
            expect_success=False,
            expected_error_substring="Input should be",
        )
    )

    return cases
GOLDEN_CASES: List[GoldenCase] = (
    _create_new_client_cases()
    + _create_new_client_negative_cases()
    + _create_new_opportunity_cases()
    + _create_new_opportunity_negative_cases()
    + _create_quote_cases()
    + _create_quote_negative_cases()
    + _upload_document_cases()
    + _upload_document_negative_cases()
    + _modify_opportunity_cases()
    + _modify_opportunity_negative_cases()
)


def cases_by_task(task: str) -> List[GoldenCase]:
    return [case for case in GOLDEN_CASES if case.task == task]


def case_ids() -> List[str]:
    return [case.case_id for case in GOLDEN_CASES]


def summary() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for case in GOLDEN_CASES:
        counts[case.task] = counts.get(case.task, 0) + 1
    return counts.copy()


__all__ = ["GoldenCase", "GOLDEN_CASES", "cases_by_task", "case_ids", "summary"]
