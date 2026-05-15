"""Exercise 4 - structured SQLite audit trail + durable checkpointer."""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel

from common.db import db_path
from hitl_agent.graph import build_graph


console = Console()


def handle_interrupt(payload: dict) -> dict:
    kind = payload["kind"]
    reviewer_id = os.environ.get("GITHUB_USER")

    if kind == "approval_request":
        console.print(
            Panel.fit(
                f"[bold]Confidence:[/bold] {payload['confidence']:.0%}\n"
                f"[dim]{payload['confidence_reasoning']}[/dim]\n\n"
                f"{payload['summary']}",
                title="Approval request",
                border_style="green",
            )
        )
        for comment in payload.get("comments", []):
            console.print(
                f"  [{comment['severity']}] {comment['file']}:{comment.get('line') or '?'} - "
                f"{comment['body']}"
            )

        choice = ""
        while choice not in {"approve", "reject", "edit"}:
            choice = console.input("\napprove/reject/edit? ").strip().lower()
        feedback = console.input("Feedback: ").strip() if choice != "approve" else ""
        return {"choice": choice, "feedback": feedback, "reviewer_id": reviewer_id}

    if kind == "escalation":
        console.print(
            Panel.fit(
                f"[bold]Confidence:[/bold] {payload['confidence']:.0%}\n"
                f"[dim]{payload['confidence_reasoning']}[/dim]\n\n"
                f"{payload['summary']}",
                title="Escalation",
                border_style="yellow",
            )
        )
        for risk in payload.get("risk_factors", []):
            console.print(f"[red]- {risk}[/red]")
        answers = {
            question: console.input(f"\nQ: {question}\nA: ").strip()
            for question in payload["questions"]
        }
        return {"answers": answers, "reviewer_id": reviewer_id}

    raise ValueError(f"Unknown interrupt kind: {kind}")


async def run(pr_url: str, thread_id: str | None) -> None:
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 - SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as checkpointer:
        await checkpointer.setup()
        app = build_graph(checkpointer)
        cfg = {"configurable": {"thread_id": thread_id}}

        result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if result.get("posted_comment_url"):
        console.print(f"comment_url  = {result['posted_comment_url']}")
    console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    parser.add_argument("--thread", help="Resume an existing thread")
    args = parser.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
