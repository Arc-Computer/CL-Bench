"""Pydantic schema models for CRM API argument validation.

These schemas enforce strict argument structures for each CRM task,
eliminating LLM hallucination of invalid fields via Curator structured outputs.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ============================================================================
# CREATE Operations
# ============================================================================

class CreateOpportunityArgs(BaseModel):
    """Arguments for create_new_opportunity."""
    name: str
    client_id: str
    amount: float
    stage: str
    probability: Optional[int] = None
    close_date: Optional[str] = None
    owner: Optional[str] = None
    notes: Optional[str] = None


class CreateClientArgs(BaseModel):
    """Arguments for create_new_client."""
    name: str
    email: str
    status: str
    industry: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    owner: Optional[str] = None


class CreateQuoteArgs(BaseModel):
    """Arguments for create_quote."""
    opportunity_id: str
    amount: float
    status: str
    version: Optional[str] = None
    valid_until: Optional[str] = None
    quote_prefix: Optional[str] = None


class CreateContactArgs(BaseModel):
    """Arguments for create_new_contact."""
    first_name: str
    last_name: str
    client_id: str
    email: Optional[str] = None
    phone: Optional[str] = None
    title: Optional[str] = None
    notes: Optional[str] = None


class CreateContractArgs(BaseModel):
    """Arguments for create_contract."""
    client_id: str
    opportunity_id: Optional[str] = None
    status: Optional[str] = None
    value: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class AddNoteArgs(BaseModel):
    """Arguments for add_note."""
    entity_type: str
    entity_id: str
    content: str
    created_by: Optional[str] = None
    created_at: Optional[str] = None


class UploadDocumentArgs(BaseModel):
    """Arguments for upload_document."""
    entity_type: str
    entity_id: str
    file_name: str
    uploaded_by: Optional[str] = None
    uploaded_at: Optional[str] = None


# ============================================================================
# MODIFY Operations
# ============================================================================

class ModifyOpportunityArgs(BaseModel):
    """Arguments for modify_opportunity."""
    opportunity_id: str
    updates: Dict[str, Any]


class ModifyClientArgs(BaseModel):
    """Arguments for modify_client."""
    client_id: str
    updates: Dict[str, Any]


class ModifyQuoteArgs(BaseModel):
    """Arguments for modify_quote."""
    quote_id: str
    updates: Dict[str, Any]


class ModifyContactArgs(BaseModel):
    """Arguments for modify_contact."""
    contact_id: str
    updates: Dict[str, Any]


# ============================================================================
# SEARCH Operations
# ============================================================================

class OpportunitySearchArgs(BaseModel):
    """Arguments for opportunity_search."""
    stage: Optional[str] = None
    client_id: Optional[str] = None
    owner: Optional[str] = None
    amount: Optional[float] = None


class ClientSearchArgs(BaseModel):
    """Arguments for client_search."""
    status: Optional[str] = None
    industry: Optional[str] = None
    owner: Optional[str] = None
    name: Optional[str] = None


class ContactSearchArgs(BaseModel):
    """Arguments for contact_search."""
    client_id: Optional[str] = None
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class QuoteSearchArgs(BaseModel):
    """Arguments for quote_search."""
    status: Optional[str] = None
    opportunity_id: Optional[str] = None
    amount: Optional[float] = None


class ContractSearchArgs(BaseModel):
    """Arguments for contract_search."""
    status: Optional[str] = None
    client_id: Optional[str] = None


class CompanySearchArgs(BaseModel):
    """Arguments for company_search."""
    type: Optional[str] = None
    industry: Optional[str] = None
    name: Optional[str] = None


class SummarizeOpportunitiesArgs(BaseModel):
    """Arguments for summarize_opportunities."""
    stage: Optional[str] = None
    owner: Optional[str] = None


# ============================================================================
# DETAILS/VIEW Operations
# ============================================================================

class ViewOpportunityDetailsArgs(BaseModel):
    """Arguments for view_opportunity_details."""
    opportunity_id: str


class OpportunityDetailsArgs(BaseModel):
    """Arguments for opportunity_details."""
    opportunity_id: str


class QuoteDetailsArgs(BaseModel):
    """Arguments for quote_details."""
    quote_id: str


# ============================================================================
# DELETE Operations
# ============================================================================

class DeleteOpportunityArgs(BaseModel):
    """Arguments for delete_opportunity."""
    opportunity_id: str


class DeleteQuoteArgs(BaseModel):
    """Arguments for delete_quote."""
    quote_id: str


class CancelQuoteArgs(BaseModel):
    """Arguments for cancel_quote."""
    quote_id: str


# ============================================================================
# OTHER Operations
# ============================================================================

class CloneOpportunityArgs(BaseModel):
    """Arguments for clone_opportunity."""
    opportunity_id: str


class CompareQuotesArgs(BaseModel):
    """Arguments for compare_quotes."""
    quote_ids: List[str]


class CompareQuoteDetailsArgs(BaseModel):
    """Arguments for compare_quote_details."""
    quote_ids: List[str]


class QuotePrefixesArgs(BaseModel):
    """Arguments for quote_prefixes."""
    # No arguments for this task
    pass


# ============================================================================
# Task-to-Schema Mapping
# ============================================================================

TASK_SCHEMA_MAP: Dict[str, type[BaseModel]] = {
    # CREATE operations
    "create_new_opportunity": CreateOpportunityArgs,
    "create_new_client": CreateClientArgs,
    "create_quote": CreateQuoteArgs,
    "create_new_contact": CreateContactArgs,
    "create_contract": CreateContractArgs,
    "add_note": AddNoteArgs,
    "upload_document": UploadDocumentArgs,
    
    # MODIFY operations
    "modify_opportunity": ModifyOpportunityArgs,
    "modify_client": ModifyClientArgs,
    "modify_quote": ModifyQuoteArgs,
    "modify_contact": ModifyContactArgs,
    
    # SEARCH operations
    "opportunity_search": OpportunitySearchArgs,
    "client_search": ClientSearchArgs,
    "contact_search": ContactSearchArgs,
    "quote_search": QuoteSearchArgs,
    "contract_search": ContractSearchArgs,
    "company_search": CompanySearchArgs,
    "summarize_opportunities": SummarizeOpportunitiesArgs,
    
    # DETAILS operations
    "view_opportunity_details": ViewOpportunityDetailsArgs,
    "opportunity_details": OpportunityDetailsArgs,
    "quote_details": QuoteDetailsArgs,
    
    # DELETE operations
    "delete_opportunity": DeleteOpportunityArgs,
    "delete_quote": DeleteQuoteArgs,
    "cancel_quote": CancelQuoteArgs,
    
    # OTHER operations
    "clone_opportunity": CloneOpportunityArgs,
    "compare_quotes": CompareQuotesArgs,
    "compare_quote_details": CompareQuoteDetailsArgs,
    "quote_prefixes": QuotePrefixesArgs,
}

