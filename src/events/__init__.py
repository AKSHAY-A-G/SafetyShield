"""SafetyShield event modeling, evidence capture, and audit logging."""

from src.events.audit import (
    AUDIT_ACTION_EVENT_CREATED,
    AUDIT_ACTION_EVIDENCE_ERROR,
    AUDIT_ACTION_EVIDENCE_SAVED,
    AuditEntry,
    AuditLogger,
)
from src.events.evidence import (
    ClipCoverage,
    EvidenceArtifacts,
    EvidenceWriter,
    compute_clip_bounds,
    verify_evidence_clip,
)
from src.events.models import SafetyEvent, generate_event_id


__all__ = [
    "AUDIT_ACTION_EVENT_CREATED",
    "AUDIT_ACTION_EVIDENCE_ERROR",
    "AUDIT_ACTION_EVIDENCE_SAVED",
    "AuditEntry",
    "AuditLogger",
    "ClipCoverage",
    "EvidenceArtifacts",
    "EvidenceWriter",
    "SafetyEvent",
    "compute_clip_bounds",
    "generate_event_id",
    "verify_evidence_clip",
]
