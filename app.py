"""Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py
"""

from __future__ import annotations

import asyncio
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_path
from hitl_agent.graph import build_graph
from hitl_agent.sessions import list_recent_sessions


load_dotenv()
st.set_page_config(page_title="HITL PR Review", layout="wide")


def _init_state() -> None:
    defaults = {
        "thread_id": None,
        "pr_url": "",
        "interrupt_payload": None,
        "final": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    async with AsyncSqliteSaver.from_conn_string(db_path()) as checkpointer:
        await checkpointer.setup()
        app = build_graph(checkpointer)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Recent sessions")
        try:
            sessions = asyncio.run(list_recent_sessions())
        except Exception as exc:
            st.caption(f"No audit sessions yet ({exc})")
            return

        if not sessions:
            st.caption("No sessions yet.")
            return

        for idx, session in enumerate(sessions):
            label = f"{session['worst_risk'].upper()} - {session['events']} events"
            st.caption(session["pr_url"])
            if st.button(label, key=f"session_{idx}_{session['thread_id']}"):
                st.session_state.thread_id = session["thread_id"]
                st.session_state.pr_url = session["pr_url"]
                st.session_state.interrupt_payload = None
                st.session_state.final = None
                st.rerun()
            st.caption(f"Last: {session['last_event']}")
            st.divider()


def render_approval_card(payload: dict) -> dict | None:
    conf = payload["confidence"]
    st.subheader(f"Approval requested - confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    if payload.get("risk_factors"):
        st.warning("Risks: " + ", ".join(payload["risk_factors"]))

    for comment in payload.get("comments", []):
        st.markdown(
            f"- **[{comment['severity']}]** "
            f"`{comment['file']}:{comment.get('line') or '?'}` - {comment['body']}"
        )

    with st.expander("Diff preview"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_area("Feedback", key="approval_feedback")
    reviewer_id = os.environ.get("GITHUB_USER")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback, "reviewer_id": reviewer_id}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback, "reviewer_id": reviewer_id}
    if col3.button("Edit"):
        return {"choice": "edit", "feedback": feedback, "reviewer_id": reviewer_id}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    conf = payload["confidence"]
    st.subheader(f"Strong escalation - confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.expander("Diff preview"):
        st.code(payload.get("diff_preview", ""), language="diff")

    with st.form("escalation"):
        answers = {
            question: st.text_area(question, key=f"answer_{idx}")
            for idx, question in enumerate(payload["questions"])
        }
        submitted = st.form_submit_button("Submit answers", type="primary")
    if submitted:
        return {"answers": answers, "reviewer_id": os.environ.get("GITHUB_USER")}
    return None


def render_final() -> None:
    final = st.session_state.final
    if final is None:
        return

    action = final.get("final_action", "?")
    comment_url = final.get("posted_comment_url")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"{action} - comment posted")
        if comment_url:
            st.link_button("View comment on GitHub", comment_url)
    elif action == "rejected":
        st.warning("Rejected - no comment posted")
    else:
        st.info(f"final_action = {action}")

    st.caption(
        f"thread_id = {st.session_state.thread_id} - "
        f"replay: `uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )


_init_state()
st.title("HITL PR Review Agent")
render_sidebar()

with st.form("start"):
    pr_url = st.text_input(
        "PR URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", type="primary")

if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR and asking the LLM"):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

payload = st.session_state.interrupt_payload
if payload is not None:
    if payload["kind"] == "approval_request":
        answer = render_approval_card(payload)
    elif payload["kind"] == "escalation":
        answer = render_escalation_card(payload)
    else:
        st.error(f"Unknown interrupt kind: {payload['kind']}")
        answer = None

    if answer is not None:
        with st.spinner("Resuming review"):
            result = asyncio.run(
                run_graph(
                    st.session_state.pr_url,
                    st.session_state.thread_id,
                    resume_value=answer,
                )
            )
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

render_final()
