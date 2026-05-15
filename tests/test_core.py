from __future__ import annotations

import unittest

from common.schemas import AUTO_APPROVE_THRESHOLD, ESCALATE_THRESHOLD, PRAnalysis, ReviewComment, risk_level_for
from hitl_agent.comments import render_comment_body, thread_marker
from hitl_agent.routing import route_decision


class CoreBehaviorTests(unittest.TestCase):
    def test_route_decision_thresholds(self) -> None:
        self.assertEqual(route_decision(AUTO_APPROVE_THRESHOLD), "auto_approve")
        self.assertEqual(route_decision(ESCALATE_THRESHOLD - 0.01), "escalate")
        self.assertEqual(route_decision(ESCALATE_THRESHOLD), "human_approval")

    def test_risk_level_matches_thresholds(self) -> None:
        self.assertEqual(risk_level_for(AUTO_APPROVE_THRESHOLD), "low")
        self.assertEqual(risk_level_for(ESCALATE_THRESHOLD - 0.01), "high")
        self.assertEqual(risk_level_for(ESCALATE_THRESHOLD), "med")

    def test_comment_rendering_includes_marker_and_context(self) -> None:
        analysis = PRAnalysis(
            summary="Adds login and sync.",
            risk_factors=["Plaintext token storage"],
            comments=[
                ReviewComment(
                    file="auth.py",
                    line=12,
                    severity="blocker",
                    body="Use a password hashing function.",
                )
            ],
            confidence=0.42,
            confidence_reasoning="Security-sensitive changes need context.",
            escalation_questions=["Why MD5?"],
        )
        state = {
            "thread_id": "thread-123",
            "analysis": analysis,
            "human_feedback": "Please mention migration risk.",
            "escalation_answers": {"Why MD5?": "It should be changed."},
        }

        body = render_comment_body(state)

        self.assertIn(thread_marker("thread-123"), body)
        self.assertIn("Adds login and sync.", body)
        self.assertIn("Plaintext token storage", body)
        self.assertIn("auth.py:12", body)
        self.assertIn("Please mention migration risk.", body)
        self.assertIn("It should be changed.", body)


if __name__ == "__main__":
    unittest.main()
