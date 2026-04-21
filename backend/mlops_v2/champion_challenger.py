from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class CanaryPolicy:
    canary_fraction: float = 0.10
    canary_days: int = 3


def route_to_challenger(request_key: str, policy: CanaryPolicy) -> bool:
    """Stable hash routing so ~10% traffic gets challenger."""
    digest = hashlib.sha256(request_key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < policy.canary_fraction


def can_auto_promote(started_at: datetime, policy: CanaryPolicy) -> bool:
    """Allow auto-promotion after 3 full days of canary window."""
    return datetime.utcnow() >= started_at + timedelta(days=policy.canary_days)
