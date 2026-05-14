"""
## demo_hitl_pipeline

**Pillar 2: Human-in-the-Loop** — AI investigates pipeline anomalies; a human
approves or redirects before any remediation action runs.

```
Anomaly Detected → AI Investigation → HITL Review Gate → Auto-Remediate | Escalate
```

**Why Astro is essential here:**
- Regulated industries (finance, healthcare) need an audit trail of human approval
  before automated actions run — Airflow makes HITL a first-class workflow primitive
- The entire review decision is captured in Airflow's metadata DB for compliance
- `@task.agent` with `enable_hitl_review=True` provides the full pause-and-wait
  UI in Astro (see example_agent_hitl_review.py); this DAG uses `@task.llm_branch`
  to demonstrate the routing logic without requiring a running HITL plugin

**Mock mode:** set `MOCK_LLM=true` in `.env` to run end-to-end with no API key.
The investigation and branching use rule-based logic on the anomaly data.
Set `MOCK_LLM=false` and add an API key to switch to real model calls.

Customers: Wix · Ramp · Recast
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from airflow.sdk import dag, task

MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------

class AnomalyReport(BaseModel):
    """Structured anomaly diagnosis produced by the AI investigation step."""

    severity: Literal["INFO", "WARNING", "CRITICAL"]
    diagnosis: str = Field(description="What went wrong and why, in plain language")
    root_cause: str = Field(description="Most likely root cause in 1-2 sentences")
    affected_dags: list[str] = Field(description="DAG IDs affected by this anomaly")
    recommended_fix: str = Field(description="Specific, actionable remediation steps")
    auto_remediable: bool = Field(
        description="True if the fix is safe to apply automatically without human judgment"
    )
    confidence: Literal["LOW", "MEDIUM", "HIGH"]


# ---------------------------------------------------------------------------
# Mock helpers — rule-based responses used when MOCK_LLM=true
# ---------------------------------------------------------------------------

def _mock_anomaly_report(anomalies: list[dict]) -> dict:
    has_oom = any(
        "OOM" in str(a.get("error_message", "")) or "memory" in str(a.get("error_message", "")).lower()
        for a in anomalies
    )
    has_scheduling = any(a.get("anomaly_type") == "SCHEDULING_DELAY" for a in anomalies)
    dag_ids = [a["dag_id"] for a in anomalies]

    severity: Literal["INFO", "WARNING", "CRITICAL"] = (
        "CRITICAL" if has_oom else "WARNING" if has_scheduling else "INFO"
    )

    return AnomalyReport(
        severity=severity,
        diagnosis=(
            "Two concurrent pipeline failures detected: ml_feature_pipeline is crashing with "
            "OOM errors on the compute_embeddings task, and customer_data_sync has stalled "
            "because it depends on feature data that never landed."
        ),
        root_cause=(
            "The compute_embeddings task exceeded its 8 GB memory ceiling when processing "
            "a larger-than-expected embedding batch. The customer_data_sync deadlock is a "
            "downstream cascade — it has queued 4 runs waiting on the blocked feature pipeline."
        ),
        affected_dags=dag_ids,
        recommended_fix=(
            "1. Increase compute_embeddings worker memory limit to 16 GB in the pool config. "
            "2. Once ml_feature_pipeline recovers, clear the 4 queued customer_data_sync runs."
        ),
        auto_remediable=not has_oom,
        confidence="HIGH",
    ).model_dump()


def _mock_review_branch(review_packet: dict) -> str:
    if review_packet.get("auto_remediable") and review_packet.get("severity") != "CRITICAL":
        return "auto_remediate"
    return "manual_escalate"


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

@dag(
    dag_id="demo_hitl_pipeline",
    description="[DEMO] Pillar 2: Human-in-the-Loop — AI diagnoses anomalies; human gates remediation",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["demo", "ai", "hitl", "human-in-the-loop", "pillar-2"],
    doc_md=__doc__,
)
def demo_hitl_pipeline():

    @task
    def detect_pipeline_anomalies() -> list[dict]:
        """Detect anomalies in running pipelines.

        In production: query the Airflow REST API, Astro Observe webhooks,
        or a monitoring system (Datadog, CloudWatch) for recent failures,
        scheduling delays, and unexpected task states.
        """
        return [
            {
                "dag_id": "customer_data_sync",
                "anomaly_type": "SCHEDULING_DELAY",
                "description": "DAG has not started for 3 consecutive scheduled intervals",
                "last_successful_run": "2026-05-10T14:00:00Z",
                "current_status": "not_started",
                "queued_run_count": 4,
                "recent_errors": [],
            },
            {
                "dag_id": "ml_feature_pipeline",
                "anomaly_type": "TASK_FAILURE",
                "description": "Task 'compute_embeddings' failing with OOM errors for 6 hours",
                "last_successful_run": "2026-05-10T08:00:00Z",
                "current_status": "failed",
                "failed_task": "compute_embeddings",
                "error_message": "Worker killed — memory limit (8 GB) exceeded",
                "retry_count": 3,
                "recent_errors": [
                    "MemoryError: Unable to allocate 12.4 GiB for array",
                    "Worker pod OOMKilled — node pressure detected",
                ],
            },
        ]

    # ------------------------------------------------------------------
    # @task.llm: AI investigation of the detected anomalies
    # ------------------------------------------------------------------
    if MOCK_LLM:
        @task
        def investigate_anomalies(anomalies: list[dict]) -> dict:
            """Mock: rule-based anomaly diagnosis from the raw anomaly data."""
            return _mock_anomaly_report(anomalies)
    else:
        @task.llm(
            llm_conn_id="pydanticai_default",
            output_type=AnomalyReport,
            system_prompt=(
                "You are an expert Airflow platform engineer and SRE. "
                "Given anomaly data from a production Airflow deployment, diagnose the root cause "
                "and recommend a specific remediation. Flag whether the fix is safe to apply "
                "automatically or requires human judgment. Be direct — an on-call engineer "
                "will use this to make a time-sensitive decision."
            ),
        )
        def investigate_anomalies(anomalies: list[dict]) -> str:
            """Build an investigation prompt from detected anomaly data."""
            lines = ["Airflow Platform Anomaly Report", "=" * 40, ""]
            for i, anomaly in enumerate(anomalies, 1):
                lines.append(f"Anomaly {i}: {anomaly['anomaly_type']}")
                lines.append(f"  DAG: {anomaly['dag_id']}")
                lines.append(f"  Description: {anomaly['description']}")
                if anomaly.get("error_message"):
                    lines.append(f"  Error: {anomaly['error_message']}")
                for err in anomaly.get("recent_errors", []):
                    lines.append(f"    - {err}")
                lines.append(f"  Last success: {anomaly.get('last_successful_run', 'unknown')}")
                lines.append("")
            lines.append(
                "Diagnose these anomalies, identify root causes, and recommend a specific fix."
            )
            return "\n".join(lines)

    @task
    def prepare_hitl_review_packet(
        anomalies: list[dict],
        report: AnomalyReport | dict,
    ) -> dict:
        """Prepare the review packet and notify the on-call reviewer.

        In production: post to Slack, page via PagerDuty, open a Jira ticket,
        then surface the full report in the Astro HITL Review UI tab so the
        reviewer can approve / reject / request changes before the pipeline continues.
        """
        if isinstance(report, dict):
            report = AnomalyReport.model_validate(report)

        print("\n" + "=" * 60)
        print("AI INVESTIGATION COMPLETE — AWAITING HUMAN REVIEW")
        print("=" * 60)
        print(f"\nSeverity:        {report.severity}")
        print(f"Confidence:      {report.confidence}")
        print(f"Auto-remediable: {report.auto_remediable}")
        print(f"\nDiagnosis:\n  {report.diagnosis}")
        print(f"\nRoot Cause:\n  {report.root_cause}")
        print(f"\nRecommended Fix:\n  {report.recommended_fix}")
        print(f"\nAffected DAGs:   {', '.join(report.affected_dags)}")
        print("\nPipeline paused. Open the HITL Review tab to approve or reject.")

        return {
            "severity": report.severity,
            "diagnosis": report.diagnosis,
            "root_cause": report.root_cause,
            "recommended_fix": report.recommended_fix,
            "affected_dags": report.affected_dags,
            "auto_remediable": report.auto_remediable,
            "confidence": report.confidence,
            "anomaly_count": len(anomalies),
            "mock_mode": MOCK_LLM,
            "review_requested_at": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # HITL Review Gate — branches to auto_remediate or manual_escalate
    #
    # Real mode: @task.llm_branch — LLM reads the report and picks a branch.
    # Mock mode: @task.branch — rule-based decision from severity + auto_remediable.
    #
    # Both return the task_id of the branch to run; the >> operator registers
    # auto_remediate and manual_escalate as downstream candidates, and the branch
    # operator skips whichever one isn't selected.
    # ------------------------------------------------------------------
    if MOCK_LLM:
        @task.branch
        def hitl_review_gate(review_packet: dict) -> str:
            """Mock: rule-based branch decision from the review packet."""
            return _mock_review_branch(review_packet)
    else:
        @task.llm_branch(
            llm_conn_id="pydanticai_default",
            system_prompt=(
                "You are simulating a human reviewer's decision on an AI pipeline anomaly report. "
                "Based on severity and auto_remediable, choose the action. "
                "Return EXACTLY one of: 'auto_remediate' or 'manual_escalate'. "
                "Choose 'auto_remediate' only if severity is WARNING or INFO and auto_remediable is true. "
                "Choose 'manual_escalate' if severity is CRITICAL or auto_remediable is false."
            ),
        )
        def hitl_review_gate(review_packet: dict) -> str:
            """Branch based on the reviewer's decision."""
            return (
                f"Severity: {review_packet['severity']} | "
                f"Auto-remediable: {review_packet['auto_remediable']} | "
                f"Confidence: {review_packet['confidence']}\n"
                f"Diagnosis: {review_packet['diagnosis']}\n"
                f"Recommended Fix: {review_packet['recommended_fix']}\n\n"
                f"Return 'auto_remediate' or 'manual_escalate'."
            )

    @task
    def auto_remediate(review_packet: dict) -> str:
        """Apply the AI-recommended fix — human approved, safe to proceed.

        In production: clear stuck DAG runs via the Airflow REST API, adjust worker
        memory limits, update pool sizes, or trigger a re-run of affected DAGs.
        Full audit trail preserved in Airflow metadata and Astro Observe.
        """
        print("\nHUMAN APPROVED — Applying auto-remediation")
        print(f"Fix:           {review_packet['recommended_fix']}")
        print(f"Affected DAGs: {', '.join(review_packet['affected_dags'])}")
        print(f"Completed at:  {datetime.utcnow().isoformat()}")
        return f"Remediation applied at {datetime.utcnow().isoformat()}"

    @task
    def manual_escalate(review_packet: dict) -> str:
        """Escalate to a human engineer — severity too high or reviewer rejected auto-fix.

        In production: create a PagerDuty incident, open a Jira ticket with full
        context, post a detailed thread in #data-oncall, freeze affected deployments.
        """
        print("\nESCALATING TO ON-CALL ENGINEER")
        print(f"Severity:  {review_packet['severity']}")
        print(f"Diagnosis: {review_packet['diagnosis']}")
        print("Actions:   PagerDuty incident created → Jira ticket opened → #data-oncall notified")
        return f"Escalation created at {datetime.utcnow().isoformat()}"

    # -----------------------------------------------------------------------
    # Pipeline wiring
    # -----------------------------------------------------------------------
    anomalies = detect_pipeline_anomalies()
    report = investigate_anomalies(anomalies=anomalies)
    review_packet = prepare_hitl_review_packet(anomalies=anomalies, report=report)

    auto_fix = auto_remediate(review_packet=review_packet)
    escalate = manual_escalate(review_packet=review_packet)
    hitl_review_gate(review_packet=review_packet) >> [auto_fix, escalate]


demo_hitl_pipeline()
