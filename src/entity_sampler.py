from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
import random

from .crm_sandbox import (
    Client,
    Contact,
    Opportunity,
    Quote,
    Contract,
    Document,
    Company,
    MockCrmApi,
    ClientStatus,
    OpportunityStage,
    QuoteStatus,
    ContractStatus,
    CompanyType,
    DocumentEntityType,
)
from .data_pools import (
    COMPANY_NAMES,
    INDUSTRIES,
    FIRST_NAMES,
    LAST_NAMES,
    OWNERS,
    JOB_TITLES,
    OPPORTUNITY_NAME_TEMPLATES,
    QUOTE_VERSION_PREFIXES,
    DOCUMENT_TYPES,
    FILE_EXTENSIONS,
    NOTE_TEMPLATES,
    STREET_NAMES,
    CITIES,
    STATES,
)


@dataclass
class SamplerConfig:
    company_size_distribution: Dict[str, float] = field(default_factory=lambda: {
        "small": 0.3,
        "medium": 0.5,
        "large": 0.2,
    })
    industry_distribution: Optional[Dict[str, float]] = None
    client_status_distribution: Dict[str, float] = field(default_factory=lambda: {
        "Active": 0.6,
        "Prospect": 0.3,
        "Inactive": 0.1,
    })
    opportunity_stage_distribution: Dict[str, float] = field(default_factory=lambda: {
        "Prospecting": 0.2,
        "Qualification": 0.2,
        "Proposal": 0.2,
        "Negotiation": 0.2,
        "Closed-Won": 0.1,
        "Closed-Lost": 0.1,
    })
    quote_status_distribution: Dict[str, float] = field(default_factory=lambda: {
        "Draft": 0.3,
        "Sent": 0.3,
        "Approved": 0.2,
        "Rejected": 0.1,
        "Canceled": 0.1,
    })
    contract_status_distribution: Dict[str, float] = field(default_factory=lambda: {
        "Active": 0.5,
        "Pending": 0.3,
        "Expired": 0.2,
    })
    company_type_distribution: Dict[str, float] = field(default_factory=lambda: {
        "Partner": 0.4,
        "Vendor": 0.3,
        "Competitor": 0.3,
    })
    min_amount: float = 1000.0
    max_amount: float = 500000.0
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)


class EntitySampler:
    def __init__(self, api: MockCrmApi, config: Optional[SamplerConfig] = None):
        self.api = api
        self.config = config or SamplerConfig()
        self._used_company_names = set()
        self._used_emails = set()

    def _weighted_choice(self, distribution: Dict[str, float]) -> str:
        choices = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(choices, weights=weights, k=1)[0]

    def _generate_email(self, first_name: str, last_name: str, company_name: str) -> str:
        domain = company_name.lower().replace(" ", "").replace(".", "")
        if len(domain) > 20:
            domain = domain[:20]
        base_email = f"{first_name.lower()}.{last_name.lower()}@{domain}.com"
        email = base_email
        counter = 1
        while email in self._used_emails:
            email = f"{first_name.lower()}.{last_name.lower()}{counter}@{domain}.com"
            counter += 1
        self._used_emails.add(email)
        return email

    def _generate_address(self) -> str:
        number = random.randint(100, 9999)
        street = random.choice(STREET_NAMES)
        city = random.choice(CITIES)
        state = random.choice(STATES)
        zip_code = random.randint(10000, 99999)
        return f"{number} {street}, {city}, {state} {zip_code}"

    def _generate_phone(self) -> str:
        area = random.randint(200, 999)
        exchange = random.randint(200, 999)
        number = random.randint(1000, 9999)
        return f"+1-{area}-{exchange}-{number}"

    def sample_company(self) -> Company:
        available_names = [n for n in COMPANY_NAMES if n not in self._used_company_names]
        if not available_names:
            name = f"Company {len(self._used_company_names) + 1}"
        else:
            name = random.choice(available_names)
        self._used_company_names.add(name)

        company_type = self._weighted_choice(self.config.company_type_distribution)
        industry = random.choice(INDUSTRIES)
        address = self._generate_address()

        return Company(
            name=name,
            type=company_type,
            industry=industry,
            address=address,
        )

    def sample_client(self) -> Client:
        available_names = [n for n in COMPANY_NAMES if n not in self._used_company_names]
        if not available_names:
            name = f"Client {len(self._used_company_names) + 1}"
        else:
            name = random.choice(available_names)
        self._used_company_names.add(name)

        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        email = self._generate_email(first_name, last_name, name)
        status = self._weighted_choice(self.config.client_status_distribution)
        industry = random.choice(INDUSTRIES)
        phone = self._generate_phone()
        address = self._generate_address()
        owner = random.choice(OWNERS)

        client = self.api.create_new_client(
            name=name,
            email=email,
            status=status,
            industry=industry,
            phone=phone,
            address=address,
            owner=owner,
        )
        return client

    def sample_contact(self, client_id: str) -> Contact:
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        client = self.api.clients[client_id]
        email = self._generate_email(first_name, last_name, client.name)
        title = random.choice(JOB_TITLES)
        phone = self._generate_phone()

        contact = self.api.create_new_contact(
            first_name=first_name,
            last_name=last_name,
            client_id=client_id,
            email=email,
            title=title,
            phone=phone,
        )
        return contact

    def sample_opportunity(self, client_id: str) -> Opportunity:
        client = self.api.clients[client_id]
        year = random.randint(2024, 2026)
        template = random.choice(OPPORTUNITY_NAME_TEMPLATES)
        name = template.format(company=client.name, year=year)

        stage = self._weighted_choice(self.config.opportunity_stage_distribution)
        amount = random.uniform(self.config.min_amount, self.config.max_amount)
        amount = round(amount, 2)

        probability_map = {
            "Prospecting": random.randint(10, 30),
            "Qualification": random.randint(25, 45),
            "Proposal": random.randint(40, 60),
            "Negotiation": random.randint(60, 85),
            "Closed-Won": 99,
            "Closed-Lost": 1,
        }
        probability = probability_map.get(stage, 50)

        days_ahead = random.randint(30, 180)
        close_date = date.today() + timedelta(days=days_ahead)
        owner = random.choice(OWNERS)

        opportunity = self.api.create_new_opportunity(
            name=name,
            client_id=client_id,
            amount=amount,
            stage=stage,
            probability=probability,
            close_date=close_date,
            owner=owner,
        )
        return opportunity

    def sample_quote(self, opportunity_id: str) -> Quote:
        opportunity = self.api.opportunities[opportunity_id]
        amount = opportunity.amount or random.uniform(self.config.min_amount, self.config.max_amount)
        amount = round(amount, 2)

        status = self._weighted_choice(self.config.quote_status_distribution)
        prefix = random.choice(QUOTE_VERSION_PREFIXES)
        version = f"{prefix}-{random.randint(1, 999):03d}"

        days_valid = random.randint(30, 90)
        valid_until = date.today() + timedelta(days=days_valid)

        quote = self.api.create_quote(
            opportunity_id=opportunity_id,
            amount=amount,
            status=status,
            version=version,
            valid_until=valid_until,
            quote_prefix=prefix,
        )
        return quote

    def sample_contract(self, client_id: str, opportunity_id: Optional[str] = None) -> Contract:
        status = self._weighted_choice(self.config.contract_status_distribution)
        value = random.uniform(self.config.min_amount, self.config.max_amount)
        value = round(value, 2)

        start_days = random.randint(-30, 30)
        start_date = date.today() + timedelta(days=start_days)

        duration = random.randint(180, 720)
        end_date = start_date + timedelta(days=duration)

        contract = self.api.create_contract(
            client_id=client_id,
            opportunity_id=opportunity_id,
            status=status,
            value=value,
            start_date=start_date,
            end_date=end_date,
        )
        return contract

    def sample_document(self, entity_type: str, entity_id: str) -> Document:
        doc_type = random.choice(DOCUMENT_TYPES)
        extension = random.choice(FILE_EXTENSIONS)
        timestamp = datetime.now().strftime("%Y%m%d")
        file_name = f"{doc_type}_{timestamp}.{extension}"

        uploaded_by = random.choice(OWNERS)

        document = self.api.upload_document(
            entity_type=entity_type,
            entity_id=entity_id,
            file_name=file_name,
            uploaded_by=uploaded_by,
            uploaded_at=datetime.now(),
        )
        return document

    def sample_note(self, entity_type: str, entity_id: str) -> Any:
        content = random.choice(NOTE_TEMPLATES)
        created_by = random.choice(OWNERS)

        note = self.api.add_note(
            entity_type=entity_type,
            entity_id=entity_id,
            content=content,
            created_by=created_by,
            created_at=datetime.now(),
        )
        return note
