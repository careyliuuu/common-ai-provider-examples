"""
## demo_inference_routing

**Pillar 3: Inference** — Scheduling, sequencing, and monitoring the model
calls that turn raw context into decisions.

```
Ingest Records → @task.llm classification (parallel) → Route by decision → Downstream systems
                                                     ↘ Monitoring LLM pass → Observability stack
```

**Why Astro is essential here:**
- Model calls must be scheduled reliably — not triggered ad-hoc
- Each inference step needs independent retry logic, timeout handling, and lineage
- Routing decisions are auditable: who decided what, when, and with what confidence
- Batch result monitoring: if the model starts over-classifying EMERGENT cases,
  Astro Observe catches it before it becomes a patient safety issue

**Mock mode:** set `MOCK_LLM=true` in `.env` to run end-to-end with no API key.
Triage uses rule-based acuity logic on patient vitals and chief complaint.
Set `MOCK_LLM=false` and add an API key to switch to real model calls.

Customers: athenahealth · Saks Fifth Avenue · popety.io
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

class ClinicalTriageResult(BaseModel):
    """Structured triage result for a single patient record."""

    patient_id: str
    acuity_level: Literal["ROUTINE", "URGENT", "EMERGENT"]
    primary_concern: str = Field(description="Primary clinical concern in plain language")
    recommended_pathway: Literal[
        "standard_intake", "expedited_review", "immediate_escalation"
    ]
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    flags: list[str] = Field(
        default_factory=list,
        description="Clinical flags for the care team (allergies, interactions, etc.)",
    )


class BatchInferenceSummary(BaseModel):
    """Monitoring summary for the inference batch."""

    total_records: int
    routine_count: int
    urgent_count: int
    emergent_count: int
    low_confidence_count: int
    processing_time_seconds: float
    model_decision_narrative: str = Field(
        description="Brief narrative of patterns observed across the batch"
    )


# ---------------------------------------------------------------------------
# Mock helpers — rule-based responses used when MOCK_LLM=true
# ---------------------------------------------------------------------------

def _mock_triage(record: dict) -> dict:
    complaint = record["chief_complaint"].lower()
    vitals = record["vitals"]
    history = " ".join(record["history"]).lower()
    meds = " ".join(record["current_medications"]).lower()

    flags: list[str] = []
    if "prior mi" in history:
        flags.append("prior MI — high cardiac risk")
    if "warfarin" in meds or "anticoagul" in history:
        flags.append("anticoagulated — bleeding risk")
    if vitals["spo2"] < 94:
        flags.append(f"SpO2 {vitals['spo2']}% — hypoxic")
    if "allerg" in history and "allerg" in complaint:
        flags.append("known allergen exposure")

    # EMERGENT: life-threatening presentations
    if (
        vitals["spo2"] < 94
        or "chest pain" in complaint
        or "worst headache" in complaint
        or ("breathing" in complaint and "allerg" in history)
    ):
        return ClinicalTriageResult(
            patient_id=record["patient_id"],
            acuity_level="EMERGENT",
            primary_concern=record["chief_complaint"],
            recommended_pathway="immediate_escalation",
            confidence="HIGH",
            flags=flags,
        ).model_dump()

    # ROUTINE: stable, non-urgent presentations
    if "follow-up" in complaint or ("sore throat" in complaint and vitals["hr"] < 80):
        return ClinicalTriageResult(
            patient_id=record["patient_id"],
            acuity_level="ROUTINE",
            primary_concern=record["chief_complaint"],
            recommended_pathway="standard_intake",
            confidence="HIGH",
            flags=flags,
        ).model_dump()

    # URGENT: needs attention within 1 hour
    return ClinicalTriageResult(
        patient_id=record["patient_id"],
        acuity_level="URGENT",
        primary_concern=record["chief_complaint"],
        recommended_pathway="expedited_review",
        confidence="MEDIUM",
        flags=flags or ["requires prompt evaluation"],
    ).model_dump()


def _mock_inference_summary(
    triage_results: list[ClinicalTriageResult | dict],
    routing: dict,
    batch_start: str,
) -> dict:
    results = [
        ClinicalTriageResult.model_validate(r) if isinstance(r, dict) else r
        for r in triage_results
    ]
    elapsed = (datetime.utcnow() - datetime.fromisoformat(batch_start)).total_seconds()
    total = len(results)
    emergent = len(routing.get("emergent", []))
    urgent = len(routing.get("urgent", []))
    routine = len(routing.get("routine", []))
    low_conf = len(routing.get("low_confidence", []))

    emergent_pct = emergent / max(total, 1)
    narrative = (
        f"Batch of {total} patients processed: {emergent} EMERGENT ({emergent_pct:.0%}), "
        f"{urgent} URGENT, {routine} ROUTINE. "
        + (
            "ALERT: emergent rate exceeds 30% threshold — verify model calibration."
            if emergent_pct > 0.30
            else "Decision distribution within expected clinical norms for an ED intake batch."
        )
    )
    return BatchInferenceSummary(
        total_records=total,
        routine_count=routine,
        urgent_count=urgent,
        emergent_count=emergent,
        low_confidence_count=low_conf,
        processing_time_seconds=round(elapsed, 2),
        model_decision_narrative=narrative,
    ).model_dump()


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

@dag(
    dag_id="demo_inference_routing",
    description="[DEMO] Pillar 3: Inference — Scheduling and routing LLM calls that turn context into decisions",
    start_date=datetime(2026, 1, 1),
    schedule="@hourly",
    catchup=False,
    tags=["demo", "ai", "inference", "routing", "pillar-3"],
    doc_md=__doc__,
)
def demo_inference_routing():

    @task
    def record_batch_start() -> str:
        """Capture the batch start timestamp for processing-time metrics."""
        return datetime.utcnow().isoformat()

    @task
    def ingest_patient_records() -> list[dict]:
        """Ingest incoming patient records for triage.

        In production: poll an HL7/FHIR API, read from an S3 landing zone,
        or consume from a Kafka topic. Airflow's @hourly schedule guarantees
        this runs on time, retries on failure, and never double-processes.
        """
        return [
            {
                "patient_id": "PT-10042",
                "age": 67,
                "chief_complaint": "Chest pain radiating to left arm, onset 45 minutes ago",
                "vitals": {"bp": "158/94", "hr": 102, "spo2": 96, "temp": 98.6},
                "history": ["Hypertension", "Type 2 Diabetes", "Prior MI in 2021"],
                "current_medications": ["Metformin", "Lisinopril", "Aspirin"],
                "arrival_mode": "ambulance",
            },
            {
                "patient_id": "PT-10043",
                "age": 34,
                "chief_complaint": "Mild sore throat and congestion for 3 days",
                "vitals": {"bp": "118/76", "hr": 72, "spo2": 99, "temp": 99.1},
                "history": ["Seasonal allergies"],
                "current_medications": ["Loratadine"],
                "arrival_mode": "walk-in",
            },
            {
                "patient_id": "PT-10044",
                "age": 8,
                "chief_complaint": "Difficulty breathing, known peanut allergy, possible exposure",
                "vitals": {"bp": "90/60", "hr": 128, "spo2": 92, "temp": 98.4},
                "history": ["Severe peanut allergy (EpiPen prescribed)", "Asthma"],
                "current_medications": ["Albuterol inhaler", "EpiPen"],
                "arrival_mode": "parent",
            },
            {
                "patient_id": "PT-10045",
                "age": 52,
                "chief_complaint": "Follow-up for blood pressure medication adjustment",
                "vitals": {"bp": "142/88", "hr": 68, "spo2": 98, "temp": 98.2},
                "history": ["Hypertension"],
                "current_medications": ["Amlodipine"],
                "arrival_mode": "scheduled",
            },
            {
                "patient_id": "PT-10046",
                "age": 78,
                "chief_complaint": "Sudden onset severe headache, worst of life, with confusion",
                "vitals": {"bp": "182/110", "hr": 88, "spo2": 97, "temp": 98.8},
                "history": ["Atrial fibrillation", "Warfarin therapy"],
                "current_medications": ["Warfarin", "Digoxin"],
                "arrival_mode": "family",
            },
        ]

    # ------------------------------------------------------------------
    # @task.llm: classify each patient record (parallel via expand)
    # ------------------------------------------------------------------
    if MOCK_LLM:
        @task
        def classify_patient_acuity(record: dict) -> dict:
            """Mock: rule-based triage from patient vitals and chief complaint."""
            return _mock_triage(record)
    else:
        @task.llm(
            llm_conn_id="pydanticai_default",
            output_type=ClinicalTriageResult,
            system_prompt=(
                "You are a clinical triage AI assistant supporting emergency department nurses. "
                "Given patient intake data, classify acuity and recommend a care pathway. "
                "Be conservative: when uncertain, escalate. "
                "Acuity levels: ROUTINE (stable, non-urgent), URGENT (needs attention within 1 hour), "
                "EMERGENT (life-threatening, immediate intervention required). "
                "IMPORTANT: This is a demonstration system. In production, all AI classifications "
                "are reviewed by licensed medical staff before any patient action is taken."
            ),
        )
        def classify_patient_acuity(record: dict) -> str:
            """Build a triage prompt for a single patient record."""
            return (
                f"Patient ID: {record['patient_id']} | Age: {record['age']}\n"
                f"Chief Complaint: {record['chief_complaint']}\n"
                f"Vitals: BP {record['vitals']['bp']}, HR {record['vitals']['hr']}, "
                f"SpO2 {record['vitals']['spo2']}%, Temp {record['vitals']['temp']}°F\n"
                f"History: {', '.join(record['history'])}\n"
                f"Medications: {', '.join(record['current_medications'])}\n"
                f"Arrival: {record['arrival_mode']}\n\n"
                f"Classify acuity and recommend a care pathway for this patient."
            )

    @task
    def route_by_acuity(triage_results: list[ClinicalTriageResult | dict]) -> dict:
        """Route patients to the appropriate downstream system based on inference results.

        Airflow guarantees this task only runs after all parallel inference calls complete,
        with full lineage showing which model call produced each routing decision.
        """
        routing: dict[str, list[str]] = {
            "emergent": [],
            "urgent": [],
            "routine": [],
            "low_confidence": [],
        }
        for r in triage_results:
            result = ClinicalTriageResult.model_validate(r) if isinstance(r, dict) else r
            if result.confidence == "LOW":
                routing["low_confidence"].append(result.patient_id)
            if result.acuity_level == "EMERGENT":
                routing["emergent"].append(result.patient_id)
            elif result.acuity_level == "URGENT":
                routing["urgent"].append(result.patient_id)
            else:
                routing["routine"].append(result.patient_id)

        print("\n" + "=" * 60)
        print("INFERENCE ROUTING COMPLETE")
        print("=" * 60)
        for category, patients in routing.items():
            if patients:
                print(f"  {category.upper():18s}: {', '.join(patients)}")
        return routing

    @task
    def trigger_emergent_response(routing: dict) -> None:
        """Trigger immediate response protocols for EMERGENT patients.

        In production: POST to the hospital EHR system, page the trauma team,
        update the ED dashboard, reserve a bay.
        """
        patients = routing.get("emergent", [])
        if not patients:
            print("No EMERGENT patients in this batch.")
            return
        print(f"\nEMERGENT RESPONSE TRIGGERED: {', '.join(patients)}")
        print("EHR updated → Trauma team paged → ED bay reserved → Family notified")

    @task
    def queue_for_expedited_review(routing: dict) -> None:
        """Queue URGENT patients for expedited nursing review.

        In production: write to priority queue in EHR, update the nursing
        dashboard, set a 1-hour follow-up reminder.
        """
        patients = routing.get("urgent", [])
        if not patients:
            print("No URGENT patients in this batch.")
            return
        print(f"\nEXPEDITED REVIEW queued: {', '.join(patients)}")
        print("EHR priority flag set → Nursing dashboard updated → 60-min timer started")

    @task
    def process_standard_intake(routing: dict) -> None:
        """Process ROUTINE patients through standard intake workflow.

        In production: write to standard intake queue, schedule nursing
        assessment, send patient portal notification with estimated wait time.
        """
        patients = routing.get("routine", [])
        if not patients:
            print("No ROUTINE patients in this batch.")
            return
        print(f"\nSTANDARD INTAKE: {', '.join(patients)}")
        print("Standard queue → Patient portal notification → Estimated wait time sent")

    @task
    def flag_for_human_review(routing: dict) -> None:
        """Flag low-confidence classifications for mandatory human review.

        Where Inference and HITL intersect: when the model isn't confident,
        it automatically surfaces the case for a human before any routing action runs.
        In production: route to 'pending clinical review' queue, page the charge nurse.
        """
        patients = routing.get("low_confidence", [])
        if not patients:
            return
        print(f"\nLOW CONFIDENCE — HUMAN REVIEW REQUIRED: {', '.join(patients)}")
        print("Routing held → Charge nurse paged → Manual review queue updated")

    # ------------------------------------------------------------------
    # @task.llm: monitoring pass — interpret model decision distribution
    # ------------------------------------------------------------------
    if MOCK_LLM:
        @task
        def generate_inference_monitoring_report(
            triage_results: list[ClinicalTriageResult | dict],
            routing: dict,
            batch_start: str,
        ) -> dict:
            """Mock: compute monitoring summary from routing counts."""
            return _mock_inference_summary(triage_results, routing, batch_start)
    else:
        @task.llm(
            llm_conn_id="pydanticai_default",
            output_type=BatchInferenceSummary,
            system_prompt=(
                "You are a healthcare data quality monitor. "
                "Given a summary of AI triage decisions for a patient batch, flag any concerning "
                "patterns — e.g., unusually high emergent rate, clustering of low-confidence scores, "
                "or unexpected distributions. This report is reviewed by the clinical informatics team."
            ),
        )
        def generate_inference_monitoring_report(
            triage_results: list[ClinicalTriageResult | dict],
            routing: dict,
            batch_start: str,
        ) -> str:
            """Build a monitoring report prompt for the inference batch."""
            results = [
                ClinicalTriageResult.model_validate(r) if isinstance(r, dict) else r
                for r in triage_results
            ]
            elapsed = (
                datetime.utcnow() - datetime.fromisoformat(batch_start)
            ).total_seconds()
            lines = [
                "Batch Inference Monitoring Report",
                "=================================",
                f"Total patients processed: {len(results)}",
                f"Emergent: {len(routing.get('emergent', []))} | "
                f"Urgent: {len(routing.get('urgent', []))} | "
                f"Routine: {len(routing.get('routine', []))}",
                f"Low confidence (held for human review): {len(routing.get('low_confidence', []))}",
                f"Batch processing time: {elapsed:.1f}s",
                "",
                "Individual classifications:",
            ]
            for r in results:
                lines.append(
                    f"  {r.patient_id}: {r.acuity_level} [{r.confidence}] — {r.primary_concern}"
                )
            lines.append(
                "\nAnalyze for concerning patterns and provide a monitoring narrative."
            )
            return "\n".join(lines)

    @task
    def publish_monitoring_metrics(report: BatchInferenceSummary | dict) -> None:
        """Publish inference monitoring metrics to the observability stack.

        In production: write to Datadog, CloudWatch, or Astro Observe custom metrics.
        Trigger alerts when emergent_rate > 30% or low_confidence_rate > 20%.
        """
        if isinstance(report, dict):
            report = BatchInferenceSummary.model_validate(report)

        total = max(report.total_records, 1)
        print("\nINFERENCE MONITORING METRICS PUBLISHED")
        print(f"  Total records:    {report.total_records}")
        print(f"  Emergent rate:    {report.emergent_count / total:.0%}")
        print(f"  Low confidence:   {report.low_confidence_count / total:.0%}")
        print(f"  Processing time:  {report.processing_time_seconds:.1f}s")
        print(f"\nNarrative: {report.model_decision_narrative}")

    # -----------------------------------------------------------------------
    # Pipeline wiring
    # -----------------------------------------------------------------------
    batch_start = record_batch_start()
    records = ingest_patient_records()

    triage_results = classify_patient_acuity.expand(record=records)
    routing = route_by_acuity(triage_results=triage_results)

    trigger_emergent_response(routing=routing)
    queue_for_expedited_review(routing=routing)
    process_standard_intake(routing=routing)
    flag_for_human_review(routing=routing)

    monitoring_report = generate_inference_monitoring_report(
        triage_results=triage_results,
        routing=routing,
        batch_start=batch_start,
    )
    publish_monitoring_metrics(report=monitoring_report)


demo_inference_routing()
