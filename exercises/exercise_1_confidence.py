"""Exercise 1 - confidence scoring + routing."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from rich.console import Console

from common.github import fetch_pr
from common.llm import get_llm
from common.schemas import PRAnalysis, ReviewState
from hitl_agent.prompts import analyze_messages
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
    with console.status("[dim]LLM thinking[/dim]"):
        analysis = llm.invoke(
            analyze_messages(title=state["pr_title"], diff=state["pr_diff"])
        )
    console.print(
        f"  [green]OK[/green] confidence={analysis.confidence:.0%}, "
        f"{len(analysis.comments)} comment(s)"
    )
    return {"analysis": analysis}


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]-> route[/cyan]")
    confidence = state["analysis"].confidence
    decision = route_decision(confidence)
    console.print(f"  [green]OK[/green] decision=[bold]{decision}[/bold] (confidence={confidence:.0%})")
    return {"decision": decision}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[green]AUTO APPROVE[/green] - high confidence, no human needed")
    return {"final_action": "auto_approved"}


def node_human_approval(state: ReviewState) -> dict:
    console.print("[yellow]HUMAN APPROVAL[/yellow] - exercise 2 pauses here")
    return {"final_action": "pending_human_approval"}


def node_escalate(state: ReviewState) -> dict:
    console.print("[red]ESCALATE[/red] - exercise 3 asks reviewer questions")
    return {"final_action": "pending_escalation"}


def build_graph():
    graph = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr),
        ("analyze", node_analyze),
        ("route", node_route),
        ("auto_approve", node_auto_approve),
        ("human_approval", node_human_approval),
        ("escalate", node_escalate),
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
    for terminal in ("auto_approve", "human_approval", "escalate"):
        graph.add_edge(terminal, END)
    return graph.compile()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 1 - confidence routing[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    final = app.invoke({"pr_url": args.pr})

    console.rule("Final")
    console.print(f"confidence = {final['analysis'].confidence:.0%}")
    console.print(f"action     = {final.get('final_action')}")


if __name__ == "__main__":
    main()
