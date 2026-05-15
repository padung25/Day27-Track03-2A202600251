"""Audit-session queries shared by the Streamlit UI."""

from __future__ import annotations

from typing import Any

from common.db import db_conn


async def list_recent_sessions(limit: int = 25) -> list[dict[str, Any]]:
    async with db_conn() as conn:
        async with conn.execute(
            """
            WITH grouped AS (
                SELECT thread_id,
                       pr_url,
                       MIN(timestamp) AS started,
                       MAX(timestamp) AS last_event,
                       MAX(
                           CASE risk_level
                               WHEN 'high' THEN 3
                               WHEN 'med' THEN 2
                               ELSE 1
                           END
                       ) AS worst_risk_rank,
                       COUNT(*) AS events
                  FROM audit_events
                 GROUP BY thread_id, pr_url
            )
            SELECT thread_id,
                   pr_url,
                   started,
                   last_event,
                   CASE worst_risk_rank
                       WHEN 3 THEN 'high'
                       WHEN 2 THEN 'med'
                       ELSE 'low'
                   END AS worst_risk,
                   events
              FROM grouped
             ORDER BY last_event DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(row) for row in rows]
