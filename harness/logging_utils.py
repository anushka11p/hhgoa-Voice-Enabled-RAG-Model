import json
import os
from datetime import datetime

LOG_PATH = "analytics/latency_log.jsonl"

def log_run(pipeline_result, query_id=None):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "query_id": query_id,
        "total_ms": pipeline_result.total_ms,
        **pipeline_result.stage_timings_ms,
        "answer_allowed": pipeline_result.allowed,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")