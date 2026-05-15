"""Production LangGraph implementation for the HITL PR review agent."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from common.db import write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)
from hitl_agent.comments import render_comment_body, thread_marker
from hitl_agent.prompts import analyze_messages, synthesize_messages
from hitl_agent.routing import route_decision


AGENT_ID = "pr-review-agent@v0.1"


def _elapsed_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


async def audit(state: ReviewState, entry: AuditEntry) -> None:
    await write_audit_event(
        thread_id=state["thread_id"],
        pr_url=state["pr_url"],
        entry=entry,
    )


def _entry(
    *,
    action: str,
    confidence: float,
    decision: str,
    execution_time_ms: int,
    reason: str | None = None,
    reviewer_id: str | None = None,
    risk_level: str | None = None,
) -> AuditEntry:
    return AuditEntry(
        agent_id=AGENT_ID,
        action=action,
        confidence=confidence,
        risk_level=risk_level or risk_level_for(confidence),
        reviewer_id=reviewer_id,
        decision=decision,
        reason=reason,
        execution_time_ms=execution_time_ms,
    )


async def node_fetch_pr(state: ReviewState) -> dict[str, Any]:
    t0 = time.monotonic()
    pr = await asyncio.to_thread(fetch_pr, state["pr_url"])
    await audit(
        state,
        _entry(
            action="fetch_pr",
            confidence=0.0,
            risk_level="med",
            decision="pending",
            reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
            execution_time_ms=_elapsed_ms(t0),
        ),
    )
    return {
        "pr_title": pr.title,
        "pr_author": pr.author,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }


async def node_analyze(state: ReviewState) -> dict[str, Any]:
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    analysis: PRAnalysis = await llm.ainvoke(
        analyze_messages(title=state["pr_title"], diff=state["pr_diff"])
    )
    await audit(
        state,
        _entry(
            action="analyze",
            confidence=analysis.confidence,
            decision="pending",
            reason=analysis.confidence_reasoning,
            execution_time_ms=_elapsed_ms(t0),
        ),
    )
    return {"analysis": analysis}


async def node_route(state: ReviewState) -> dict[str, Any]:
    t0 = time.monotonic()
    analysis = state["analysis"]
    decision = route_decision(analysis.confidence)
    audit_decision = "escalate" if decision == "escalate" else "pending"
    if decision == "auto_approve":
        audit_decision = "auto"
    await audit(
        state,
        _entry(
            action="route",
            confidence=analysis.confidence,
            decision=audit_decision,
            reason=f"Routed to {decision}",
            execution_time_ms=_elapsed_ms(t0),
        ),
    )
    return {"decision": decision}


async def node_human_approval_pending(state: ReviewState) -> dict[str, Any]:
    t0 = time.monotonic()
    analysis = state["analysis"]
    await audit(
        state,
        _entry(
            action="human_approval_pending",
            confidence=analysis.confidence,
            decision="pending",
            reason=analysis.confidence_reasoning,
            execution_time_ms=_elapsed_ms(t0),
        ),
    )
    return {}


async def node_human_approval(state: ReviewState) -> dict[str, Any]:
    t0 = time.monotonic()
    analysis = state["analysis"]
    response = interrupt(
        {
            "kind": "approval_request",
            "pr_url": state["pr_url"],
            "confidence": analysis.confidence,
            "confidence_reasoning": analysis.confidence_reasoning,
            "summary": analysis.summary,
            "risk_factors": analysis.risk_factors,
            "comments": [comment.model_dump() for comment in analysis.comments],
            "diff_preview": state["pr_diff"][:4000],
        }
    )

    choice = response.get("choice")
    feedback = response.get("feedback") or ""
    reviewer_id = response.get("reviewer_id") or os.environ.get("GITHUB_USER")
    await audit(
        state,
        _entry(
            action="human_approval_resumed",
            confidence=analysis.confidence,
            decision=choice or "pending",
            reviewer_id=reviewer_id,
            reason=feedback or f"Reviewer chose {choice}",
            execution_time_ms=_elapsed_ms(t0),
        ),
    )
    return {
        "human_choice": choice,
        "human_feedback": feedback,
        "reviewer_id": reviewer_id,
    }


async def node_escalate_pending(state: ReviewState) -> dict[str, Any]:
    t0 = time.monotonic()
    analysis = state["analysis"]
    await audit(
        state,
        _entry(
            action="escalate_pending",
            confidence=analysis.confidence,
            decision="escalate",
            reason=analysis.confidence_reasoning,
            execution_time_ms=_elapsed_ms(t0),
        ),
    )
    return {}


async def node_escalate(state: ReviewState) -> dict[str, Any]:
    t0 = time.monotonic()
    analysis = state["analysis"]
    questions = analysis.escalation_questions[:4] or [
        "What is the intended production behavior of this PR?",
        "Are there security, migration, or compatibility constraints the diff does not show?",
    ]
    response = interrupt(
        {
            "kind": "escalation",
            "pr_url": state["pr_url"],
            "confidence": analysis.confidence,
            "confidence_reasoning": analysis.confidence_reasoning,
            "summary": analysis.summary,
            "risk_factors": analysis.risk_factors,
            "questions": questions,
            "diff_preview": state["pr_diff"][:4000],
        }
    )
    answers = response.get("answers", response)
    reviewer_id = response.get("reviewer_id") if isinstance(response, dict) else None
    reviewer_id = reviewer_id or os.environ.get("GITHUB_USER")
    await audit(
        state,
        _entry(
            action="escalate_resumed",
            confidence=analysis.confidence,
            decision="escalate",
            reviewer_id=reviewer_id,
            reason=f"Answered {len(answers)} escalation question(s)",
            execution_time_ms=_elapsed_ms(t0),
        ),
    )
    return {"escalation_answers": answers, "reviewer_id": reviewer_id}


def _reviewer_context(state: ReviewState) -> str:
    if state.get("escalation_answers"):
        return "\n".join(
            f"Q: {question}\nA: {answer}"
            for question, answer in state["escalation_answers"].items()
        )
    return f"Reviewer requested edit with feedback:\n{state.get('human_feedback') or ''}"


async def node_synthesize(state: ReviewState) -> dict[str, Any]:
    t0 = time.monotonic()
    previous = state["analysis"]
    llm = get_llm().with_structured_output(PRAnalysis)
    refined: PRAnalysis = await llm.ainvoke(
        synthesize_messages(
            diff=state["pr_diff"],
            initial_summary=previous.summary,
            initial_reasoning=previous.confidence_reasoning,
            reviewer_context=_reviewer_context(state),
        )
    )
    decision = "escalate" if state.get("escalation_answers") else "edit"
    await audit(
        state,
        _entry(
            action="synthesize",
            confidence=refined.confidence,
            decision=decision,
            reviewer_id=state.get("reviewer_id"),
            reason=refined.confidence_reasoning,
            execution_time_ms=_elapsed_ms(t0),
        ),
    )
    return {"analysis": refined}


async def _post(state: ReviewState) -> tuple[str, str | None, str | None]:
    try:
        marker = thread_marker(state["thread_id"])
        body = render_comment_body(state)
        url = await asyncio.to_thread(
            post_review_comment,
            state["pr_url"],
            body,
            marker=marker,
        )
        return "committed", url, None
    except Exception as exc:
        return "commit_failed", None, str(exc)


async def node_commit(state: ReviewState) -> dict[str, Any]:
    t0 = time.monotonic()
    analysis = state["analysis"]
    choice = state.get("human_choice")
    should_post = bool(state.get("escalation_answers")) or choice in {"approve", "edit"}
    if should_post:
        action, posted_url, error = await _post(state)
    else:
        action, posted_url, error = "rejected", None, None

    await audit(
        state,
        _entry(
            action="commit",
            confidence=analysis.confidence,
            decision=choice or ("escalate" if state.get("escalation_answers") else "pending"),
            reviewer_id=state.get("reviewer_id"),
            reason=error or f"final_action={action}",
            execution_time_ms=_elapsed_ms(t0),
        ),
    )
    return {
        "final_action": action,
        "posted_comment_body": render_comment_body(state) if should_post else None,
        "posted_comment_url": posted_url,
    }


async def node_auto_approve(state: ReviewState) -> dict[str, Any]:
    t0 = time.monotonic()
    analysis = state["analysis"]
    action, posted_url, error = await _post(state)
    await audit(
        state,
        _entry(
            action="auto_approve",
            confidence=analysis.confidence,
            decision="auto",
            reason=error or f"final_action=auto_{action}",
            execution_time_ms=_elapsed_ms(t0),
        ),
    )
    return {
        "final_action": f"auto_{action}",
        "posted_comment_body": render_comment_body(state),
        "posted_comment_url": posted_url,
    }


def _after_human(state: ReviewState) -> str:
    return "synthesize" if state.get("human_choice") == "edit" else "commit"


def build_graph(checkpointer):
    graph = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr),
        ("analyze", node_analyze),
        ("route", node_route),
        ("auto_approve", node_auto_approve),
        ("human_approval_pending", node_human_approval_pending),
        ("human_approval", node_human_approval),
        ("escalate_pending", node_escalate_pending),
        ("escalate", node_escalate),
        ("synthesize", node_synthesize),
        ("commit", node_commit),
    ]:
        graph.add_node(name, fn)

    graph.add_edge(START, "fetch_pr")
    graph.add_edge("fetch_pr", "analyze")
    graph.add_edge("analyze", "route")
    graph.add_conditional_edges(
        "route",
        lambda state: state["decision"],
        {
            "auto_approve": "auto_approve",
            "human_approval": "human_approval_pending",
            "escalate": "escalate_pending",
        },
    )
    graph.add_edge("auto_approve", END)
    graph.add_edge("human_approval_pending", "human_approval")
    graph.add_conditional_edges(
        "human_approval",
        _after_human,
        {"synthesize": "synthesize", "commit": "commit"},
    )
    graph.add_edge("escalate_pending", "escalate")
    graph.add_edge("escalate", "synthesize")
    graph.add_edge("synthesize", "commit")
    graph.add_edge("commit", END)
    return graph.compile(checkpointer=checkpointer)
