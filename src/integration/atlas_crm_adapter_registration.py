"""Register CRM harness adapter with Atlas SDK."""

from __future__ import annotations

from atlas.connectors.registry import AgentAdapter, AdapterError, register_adapter
from atlas.config.models import AdapterType, AdapterUnion, CrmHarnessAdapterConfig

from .atlas_crm_adapter import handle_crm_adapter_request


class CrmHarnessAdapter(AgentAdapter):
    """Adapter that routes CRM conversations through the harness."""

    supports_structured_payloads = True

    def __init__(self, config: CrmHarnessAdapterConfig):
        self._config = config

    async def ainvoke(self, prompt: str, metadata: dict[str, Any] | None = None) -> str:
        """Invoke the CRM harness adapter."""
        # Pass backend and use_llm_judge from config to the adapter
        call_metadata = metadata or {}
        call_metadata["backend"] = self._config.backend
        call_metadata["use_llm_judge"] = self._config.use_llm_judge
        
        # handle_crm_adapter_request is synchronous, so we call it directly
        result = handle_crm_adapter_request(
            prompt=prompt,
            metadata=call_metadata,
        )
        return result


def _build_crm_harness_adapter(config: AdapterUnion) -> AgentAdapter:
    """Build a CRM harness adapter from config."""
    if not isinstance(config, CrmHarnessAdapterConfig):
        raise AdapterError("CRM harness adapter requires CrmHarnessAdapterConfig")
    return CrmHarnessAdapter(config)


# Register the adapter when this module is imported
register_adapter(AdapterType.CRM_HARNESS, _build_crm_harness_adapter)

__all__ = ["CrmHarnessAdapter"]

