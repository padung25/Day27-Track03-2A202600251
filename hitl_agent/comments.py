"""Markdown rendering for GitHub review comments."""

from __future__ import annotations

from common.schemas import ReviewState


def thread_marker(thread_id: str) -> str:
    return f"<!-- hitl-pr-review thread_id={thread_id} -->"


def render_comment_body(state: ReviewState) -> str:
    analysis = state["analysis"]
    marker = thread_marker(state["thread_id"])
    lines = [
        marker,
        f"### Automated PR review (confidence {analysis.confidence:.0%})",
        "",
        analysis.summary,
        "",
    ]

    if analysis.risk_factors:
        lines.append("**Risk factors**")
        for risk in analysis.risk_factors:
            lines.append(f"- {risk}")
        lines.append("")

    if analysis.comments:
        lines.append("**Review comments**")
        for comment in analysis.comments:
            location = f"{comment.file}:{comment.line or '?'}"
            lines.append(f"- **[{comment.severity}]** `{location}` - {comment.body}")
        lines.append("")

    if state.get("human_feedback"):
        lines.append(f"_Reviewer note: {state['human_feedback']}_")
        lines.append("")

    if state.get("escalation_answers"):
        lines.append("_Reviewer answered escalation questions:_")
        for question, answer in state["escalation_answers"].items():
            lines.append(f"> **{question}** {answer}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
