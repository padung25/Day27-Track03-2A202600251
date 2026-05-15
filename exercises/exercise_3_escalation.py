"""Exercise 3 - escalation branch with reviewer Q&A."""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import PRAnalysis, ReviewState
from hitl_agent.comments import render_comment_body
from hitl_agent.prompts import analyze_messages, synthesize_messages
from hitl_agent.routing import route_decision


console = Console()


def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]-> fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]OK[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }


def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]-> analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff[/dim]"):
        analysis = llm.invoke(
            analyze_messages(title=state["pr_title"], diff=state["pr_diff"])
        )
    console.print(
        f"  [green]OK[/green] confidence={analysis.confidence:.0%}, "
        f"{len(analysis.escalation_questions)} escalation question(s)"
    )
    return {"analysis": analysis}


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]-> route[/cyan]")
    confidence = state["analysis"].confidence
    decision = route_decision(confidence)
    console.print(f"  [green]OK[/green] decision=[bold]{decision}[/bold] (confidence={confidence:.0%})")
    return {"decision": decision}


def node_escalate(state: ReviewState) -> dict:
    analysis = state["analysis"]
    questions = analysis.escalation_questions[:4] or [
        "What is the intended production behavior of this PR?",
        "Are there security, migration, or compatibility constraints the diff does not show?",
    ]
    answers = interrupt(
        {
            "kind": "escalation",
            "pr_url": state["pr_url"],
            "confidence": analysis.confidence,
            "confidence_reasoning": analysis.confidence_reasoning,
            "summary": analysis.summary,
            "risk_factors": analysis.risk_factors,
            "questions": questions,
            "diff_preview": state["pr_diff"][:2000],
        }
    )
    return {"escalation_answers": answers.get("answers", answers)}


def node_synthesize(state: ReviewState) -> dict:
    console.print("[cyan]-> synthesize[/cyan]")
    answers = state.get("escalation_answers") or {}
    qa = "\n".join(f"Q: {question}\nA: {answer}" for question, answer in answers.items())
    previous = state["analysis"]
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers[/dim]"):
        refined = llm.invoke(
            synthesize_messages(
                diff=state["pr_diff"],
                initial_summary=previous.summary,
                initial_reasoning=previous.confidence_reasoning,
                reviewer_context=qa,
            )
        )
    console.print(f"  [green]OK[/green] refined confidence={refined.confidence:.0%}")
    return {"analysis": refined}


def node_human_approval(state: ReviewState) -> dict:
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
            "diff_preview": state["pr_diff"][:2000],
        }
    )
    return {"human_choice": response.get("choice"), "human_feedback": response.get("feedback") or ""}


def _post(state: ReviewState, label: str) -> str:
    try:
        post_review_comment(state["pr_url"], render_comment_body(state))
        console.print(f"  [green]OK[/green] posted comment to {state['pr_url']}")
        return label
    except Exception as exc:
        console.print(f"  [red]FAILED[/red] post failed: {exc}")
        return "commit_failed"


def node_commit(state: ReviewState) -> dict:
    console.print("[cyan]-> commit[/cyan]")
    if state.get("escalation_answers"):
        return {"final_action": _post(state, "committed_after_escalation")}
    if state.get("human_choice") in {"approve", "edit"}:
        return {"final_action": _post(state, "committed")}
    console.print(f"  [yellow]skip[/yellow] no comment posted (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[cyan]-> auto_approve[/cyan] [dim]high confidence - posting directly[/dim]")
    return {"final_action": _post(state, "auto_approved")}


def build_graph():
    graph = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr),
        ("analyze", node_analyze),
        ("route", node_route),
        ("auto_approve", node_auto_approve),
        ("human_approval", node_human_approval),
        ("commit", node_commit),
        ("escalate", node_escalate),
        ("synthesize", node_synthesize),
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
            "human_approval": "human_approval",
            "escalate": "escalate",
        },
    )
    graph.add_edge("auto_approve", END)
    graph.add_edge("human_approval", "commit")
    graph.add_edge("commit", END)
    graph.add_edge("escalate", "synthesize")
    graph.add_edge("synthesize", "commit")
    return graph.compile(checkpointer=MemorySaver())


def handle_interrupt(payload: dict) -> dict:
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(
            Panel.fit(
                payload["summary"],
                title=f"Approve? conf={payload['confidence']:.0%}",
                border_style="green",
            )
        )
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    if kind == "escalation":
        console.print(
            Panel.fit(
                payload["summary"],
                title=f"Escalation conf={payload['confidence']:.0%}",
                border_style="yellow",
            )
        )
        return {question: console.input(f"Q: {question}\nA: ").strip() for question in payload["questions"]}
    raise ValueError(kind)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 3 - escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume=handle_interrupt(result["__interrupt__"][0].value)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")


if __name__ == "__main__":
    main()
