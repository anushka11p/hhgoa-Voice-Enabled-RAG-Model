"""Per-run structured logging that feeds the latency analytics.

One JSON object per pipeline run, appended to analytics/latency_log.jsonl.
Records the stage timings, which guardrail (if any) fired, and whether the run
stayed inside the core budget -- so the deployed system keeps producing the
same telemetry the offline benchmark reports on.
"""
import json
import os
from datetime import datetime, timezone

import config

LOG_PATH = str(config.LATENCY_LOG_PATH)


def log_run(pipeline_result, query_id=None, source: str = "api") -> None:
    """Append one run. Never raises -- logging must not break a served request."""
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_id": query_id,
            "source": source,
            "total_ms": round(pipeline_result.total_ms, 3),
            "core_ms": round(pipeline_result.core_ms, 3),
            "within_budget": pipeline_result.within_budget,
            **pipeline_result.stage_timings_ms,
            "answer_allowed": pipeline_result.allowed,
            "guardrail_stage": pipeline_result.guardrail_stage,
            "generation_mode": pipeline_result.generation_mode,
            "reranked": pipeline_result.reranked,
            "n_citations": len(pipeline_result.citations),
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass


def read_percentiles(limit: int = 500) -> dict:
    """Summarize the tail of the log, for the /api/metrics endpoint."""
    if not os.path.exists(LOG_PATH):
        return {"runs": 0}
    rows = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows = [r for r in rows[-limit:] if isinstance(r.get("core_ms"), (int, float))]
    if not rows:
        return {"runs": 0}

    def pct(vals, p):
        s = sorted(vals)
        if p >= 100:
            return round(s[-1], 2)
        k = max(0, min(len(s) - 1, int(round(p / 100.0 * len(s) + 0.5)) - 1))
        return round(s[k], 2)

    core = [r["core_ms"] for r in rows]
    return {
        "runs": len(rows),
        "core_ms": {"p50": pct(core, 50), "p70": pct(core, 70), "p100": pct(core, 100)},
        "within_budget_pct": round(
            100 * sum(1 for r in rows if r.get("within_budget")) / len(rows), 1
        ),
        "refusal_rate": round(
            sum(1 for r in rows if r.get("answer_allowed") is False) / len(rows), 4
        ),
        "budget_ms": config.CORE_BUDGET_MS,
    }
