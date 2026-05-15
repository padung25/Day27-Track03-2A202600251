"""Confidence routing helpers with no runtime graph dependencies."""

from __future__ import annotations

from common.schemas import AUTO_APPROVE_THRESHOLD, ESCALATE_THRESHOLD, Decision


def route_decision(confidence: float) -> Decision:
    if confidence >= AUTO_APPROVE_THRESHOLD:
        return "auto_approve"
    if confidence < ESCALATE_THRESHOLD:
        return "escalate"
    return "human_approval"
