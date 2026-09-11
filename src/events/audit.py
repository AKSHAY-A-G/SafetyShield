"""Prototype append-only audit trail logging for SafetyShield events (EVD-011).

Note on Prototype Limitations:
    This audit logging system is a lightweight software prototype designed for local
    traceability and verification during development. It writes plain JSON Lines (.jsonl)
    records to the filesystem. It is NOT tamper-proof, forensically certified,
    regulatory-compliant, or immutable storage. Anyone with filesystem write access
    can modify or delete records. Production deployments will require secure, append-only,
    or cryptographically signed audit infrastructure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


AUDIT_ACTION_EVENT_CREATED = "EVENT_CREATED"
AUDIT_ACTION_EVIDENCE_SAVED = "EVIDENCE_SAVED"
AUDIT_ACTION_EVIDENCE_ERROR = "EVIDENCE_ERROR"


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """A single audit entry formatted for JSON Lines persistence."""

    event_id: str
    camera_id: str
    session_id: str
    module_id: str
    action: str
    details: Mapping[str, Any] = field(default_factory=dict)
    audit_timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not self.camera_id:
            raise ValueError("camera_id must not be empty")
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.module_id:
            raise ValueError("module_id must not be empty")
        if not self.action:
            raise ValueError("action must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary representation."""
        return asdict(self)

    def to_json_line(self) -> str:
        """Serialize as a single-line JSON string without formatting line breaks."""
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)


class AuditLogger:
    """Appends structured audit entries to a camera/session audit.jsonl log."""

    def __init__(self, log_path: Path | str) -> None:
        self.log_path = Path(log_path).resolve()

    def log(
        self,
        event_id: str,
        camera_id: str,
        session_id: str,
        module_id: str,
        action: str,
        details: Mapping[str, Any] | None = None,
    ) -> AuditEntry:
        """Create and append an audit record to the log file."""
        entry = AuditEntry(
            event_id=event_id,
            camera_id=camera_id,
            session_id=session_id,
            module_id=module_id,
            action=action,
            details=dict(details or {}),
        )
        self.append_entry(entry)
        return entry

    def append_entry(self, entry: AuditEntry) -> None:
        """Append an existing AuditEntry to the log file atomically."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = entry.to_json_line() + "\n"
        with self.log_path.open(mode="a", encoding="utf-8") as file:
            file.write(line)

    def read_entries(self) -> list[dict[str, Any]]:
        """Read and parse all entries from the audit log."""
        if not self.log_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with self.log_path.open(mode="r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                clean = line.strip()
                if not clean:
                    continue
                try:
                    entries.append(json.loads(clean))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Malformed JSON on line {line_number} of {self.log_path}: {error}"
                    ) from error
        return entries
