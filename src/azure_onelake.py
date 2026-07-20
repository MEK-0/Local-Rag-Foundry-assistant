"""
OneLake (Microsoft Fabric) as an alternative document source for cloud mode.

STATUS: Design stub only - not implemented, not tested against a live
Fabric workspace. Evaluated as an alternative to azure_storage.py's Blob
Storage sync: Azure AI Search supports a OneLake indexer that can index
files directly from a Fabric Lakehouse (Lakehouse URL + folder path),
without a separate Blob Storage sync step.

Why this is a stub and not a real integration:
  - OneLake requires a provisioned Microsoft Fabric capacity, which is a
    separate resource from Azure AI Search/Storage/OpenAI - it has its
    own provisioning flow and regional availability.
  - At the time this stub was written, Azure AI Search itself could not
    be provisioned under the available Azure for Students subscription
    due to regional capacity limits (see README's Current limitations).
    Fabric capacity was not attempted for the same reason - attempting
    a second, likely more restricted resource type before the first one
    is confirmed available was not a good use of time.

This module exists to record the design decision and the integration
point, not to claim a working implementation. Do not call these
functions expecting them to work - they will not until implemented.
"""

from typing import List

from src.config import settings


def _require_onelake_config() -> None:
    raise NotImplementedError(
        "OneLake integration is a design stub, not implemented. "
        "See this module's docstring for why, and README's Current "
        "limitations for the underlying Azure capacity constraint."
    )


def get_onelake_lakehouse_url() -> str:
    """
    Intended to return the configured Fabric Lakehouse URL
    (e.g. https://onelake.dfs.fabric.microsoft.com/<workspace>/<lakehouse>.Lakehouse/Files)
    once ONELAKE_LAKEHOUSE_URL is added to config.py. Not implemented.
    """
    _require_onelake_config()


def list_onelake_documents(folder_path: str) -> List[str]:
    """
    Intended to list documents under a given OneLake folder path, as an
    alternative source to azure_storage.sync_documents_directory()'s
    local-filesystem-to-Blob-Storage sync. Not implemented.
    """
    _require_onelake_config()


def configure_onelake_indexer() -> None:
    """
    Intended to configure Azure AI Search's OneLake indexer against a
    Lakehouse Files path, as an alternative to azure_search.py's
    upload_chunks() (which pushes pre-chunked, pre-embedded documents
    directly). A OneLake indexer would instead let Azure AI Search pull
    and index raw files itself. Not implemented - would require design
    decisions about how this reconciles with this project's own
    node-tree parsing/chunking (see README's Local & cloud strategy)
    rather than Azure AI Search's built-in document cracking.
    """
    _require_onelake_config()