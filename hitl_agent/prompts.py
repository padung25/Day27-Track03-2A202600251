"""Prompt builders for the PR review agent."""

from __future__ import annotations

from common.schemas import AUTO_APPROVE_THRESHOLD, ESCALATE_THRESHOLD, PRAnalysis


ANALYZE_SYSTEM_PROMPT = f"""You are a senior software engineer reviewing a pull request.
Return only structured output matching {PRAnalysis.__name__}.

Review goals:
- Identify correctness, security, data integrity, migration, and test risks.
- Prefer actionable review comments over generic summaries.
- Calibrate confidence honestly from 0.0 to 1.0.
- Use confidence >= {AUTO_APPROVE_THRESHOLD:.2f} only for small, low-risk changes.
- Use confidence < {ESCALATE_THRESHOLD:.2f} for changes needing human context.
- If confidence < {ESCALATE_THRESHOLD:.2f}, populate escalation_questions with 2-4
  specific questions tied to files, code paths, or product intent.
"""


SYNTHESIZE_SYSTEM_PROMPT = """You are refining an automated PR review after human input.
Return only structured output matching PRAnalysis.

Use the original diff, the initial analysis, and reviewer context to produce the
final review that should be posted to GitHub. Keep useful findings, remove
obsolete uncertainty, and update confidence based on the added context.
"""


def analyze_messages(*, title: str, diff: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ANALYZE_SYSTEM_PROMPT},
        {"role": "user", "content": f"PR title: {title}\n\nUnified diff:\n{diff}"},
    ]


def synthesize_messages(
    *,
    diff: str,
    initial_summary: str,
    initial_reasoning: str,
    reviewer_context: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYNTHESIZE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Initial summary:\n{initial_summary}\n\n"
                f"Initial confidence reasoning:\n{initial_reasoning}\n\n"
                f"Reviewer context:\n{reviewer_context}\n\n"
                f"Unified diff:\n{diff}"
            ),
        },
    ]
