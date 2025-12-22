"""Offline evaluation harness for retrieval and synthesis."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from apps.api.services.intent import classify_legal_object
from apps.api.services.query_expansion import expand_queries
from apps.api.services.router import route_domain
from apps.api.services.rerank import RERANK_ENABLED, RERANKER_MODEL, RERANK_TOP_N
from apps.api.services.retrieval import get_legal_index_info, retrieve_citations_multi
from apps.api.services.retrieval_repo import retrieve_repo_citations
from apps.api.services.synthesis import TWO_PASS_ENABLED, synthesize_answer_grounded
from apps.api.services.synthesis_repo import synthesize_repo_answer_grounded
from apps.api.services.grounding_verify import verify_grounding

# =============================
# Data models and configuration
# =============================
DEFAULT_CONFIG = {
    "min_legal_must_include_hit_rate": 0.75,
    "min_system_must_include_hit_rate": 0.80,
    "max_grounding_failures_total": 0,
    "max_avg_retrieval_latency_ms": 1500,
}

LEGAL_PASS_THRESHOLD = 0.75
SYSTEM_PASS_THRESHOLD = 0.80

EVAL_DIR = Path("data/eval")
RUN_DIR = EVAL_DIR / "runs"
LEGAL_FILE = EVAL_DIR / "questions_legal.jsonl"
SYSTEM_FILE = EVAL_DIR / "questions_system.jsonl"


@dataclass
class QuestionItem:
    id: str
    domain: str
    question: str
    legal_object_hint: Optional[str] = None
    must_include_strings: List[str] = field(default_factory=list)
    must_include_citation_ids: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    notes: Optional[str] = None


# =====================
# Utility functionality
# =====================
def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return dict(DEFAULT_CONFIG)
    try:
        with config_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        return {**DEFAULT_CONFIG, **loaded}
    except Exception:
        return dict(DEFAULT_CONFIG)


def _load_questions(path: Path) -> List[QuestionItem]:
    items: List[QuestionItem] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(
                QuestionItem(
                    id=data.get("id"),
                    domain=data.get("domain"),
                    question=data.get("question"),
                    legal_object_hint=data.get("legal_object_hint"),
                    must_include_strings=data.get("must_include_strings", []) or [],
                    must_include_citation_ids=data.get("must_include_citation_ids", []) or [],
                    expected_keywords=data.get("expected_keywords", []) or [],
                    notes=data.get("notes"),
                )
            )
    return items


def _concat_citation_texts(citations: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for c in citations:
        parts.extend(
            [
                c.get("verbatim", ""),
                c.get("snippet", ""),
                c.get("context_before", ""),
                c.get("context_after", ""),
            ]
        )
    return "\n".join(parts).lower()


def _must_include_hits(strings: List[str], citations: List[Dict[str, Any]]) -> int:
    if not strings:
        return 0
    blob = _concat_citation_texts(citations)
    return sum(1 for s in strings if s.lower() in blob)


def _source_coverage(citations: List[Dict[str, Any]]) -> List[str]:
    sources = {c.get("source_file", "") for c in citations if c.get("source_file")}
    return sorted(sources)


# =====================
# Evaluation processing
# =====================
@dataclass
class QuestionResult:
    question: QuestionItem
    domain: str
    legal_object: Optional[str]
    queries: List[str]
    citations: List[Dict[str, Any]]
    status: str
    metrics: Dict[str, Any]
    answer: Optional[str] = None
    grounding_ok: Optional[bool] = None
    grounding_failures_count: int = 0


def _evaluate_question(item: QuestionItem, mode: str) -> QuestionResult:
    start_time = time.perf_counter()
    domain = route_domain(item.question)
    queries: List[str] = []
    citations: List[Dict[str, Any]] = []
    legal_object: Optional[str] = None
    retrieval_latency_ms = 0.0
    status = "ok"

    if domain == "CLARIFY":
        return QuestionResult(
            question=item,
            domain=domain,
            legal_object=None,
            queries=[],
            citations=[],
            status="skipped_clarify",
            metrics={},
        )

    # Retrieval stage
    retrieval_start = time.perf_counter()
    if domain == "LEGAL":
        legal_object = item.legal_object_hint
        if not legal_object:
            intent = classify_legal_object(item.question)
            legal_object = intent.legal_object
        queries = expand_queries(item.question, legal_object)
        citations = retrieve_citations_multi(queries, top_k=8, legal_object=legal_object)
    elif domain == "SYSTEM_HELP":
        citations = retrieve_repo_citations(item.question, top_k=6)
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

    # Metrics from retrieval
    must_include_hit_count = _must_include_hits(item.must_include_strings, citations)
    metrics: Dict[str, Any] = {
        "top_k_returned": len(citations),
        "must_include_hit_count": must_include_hit_count,
        "source_coverage": _source_coverage(citations),
        "latency_ms_retrieval": retrieval_latency_ms,
    }

    answer: Optional[str] = None
    grounding_ok: Optional[bool] = None
    grounding_failures_count = 0
    synthesis_latency_ms = None

    if mode == "full":
        synth_start = time.perf_counter()
        if domain == "LEGAL":
            answer = synthesize_answer_grounded(item.question, citations)
            grounding_ok, grounding_failures = verify_grounding(answer, citations)
            grounding_failures_count = len(grounding_failures)
        elif domain == "SYSTEM_HELP":
            answer = synthesize_repo_answer_grounded(item.question, citations)
            grounding_ok = True
            grounding_failures_count = 0
        synthesis_latency_ms = (time.perf_counter() - synth_start) * 1000.0
        metrics.update(
            {
                "answer_length_chars": len(answer or ""),
                "grounding_ok": grounding_ok,
                "grounding_failures_count": grounding_failures_count,
                "latency_ms_synthesis": synthesis_latency_ms,
                "latency_ms_total": (time.perf_counter() - start_time) * 1000.0,
            }
        )

    return QuestionResult(
        question=item,
        domain=domain,
        legal_object=legal_object,
        queries=queries,
        citations=citations,
        status=status,
        metrics=metrics,
        answer=answer,
        grounding_ok=grounding_ok,
        grounding_failures_count=grounding_failures_count,
    )


# =====================
# Aggregation and gates
# =====================
def _summarize(results: List[QuestionResult], mode: str) -> Dict[str, Any]:
    legal_with_expectations = [r for r in results if r.domain == "LEGAL" and r.question.must_include_strings]
    system_with_expectations = [r for r in results if r.domain == "SYSTEM_HELP" and r.question.must_include_strings]

    def _hit_rate(items: List[QuestionResult], threshold: float) -> float:
        if not items:
            return 1.0
        passes = 0
        for r in items:
            needed = len(r.question.must_include_strings)
            hits = r.metrics.get("must_include_hit_count", 0)
            if needed == 0:
                continue
            if hits == needed or (needed > 0 and hits / needed >= threshold):
                passes += 1
        return passes / len(items)

    legal_hit_rate = _hit_rate(legal_with_expectations, LEGAL_PASS_THRESHOLD)
    system_hit_rate = _hit_rate(system_with_expectations, SYSTEM_PASS_THRESHOLD)

    retrieval_latencies = [r.metrics.get("latency_ms_retrieval") for r in results if r.metrics.get("latency_ms_retrieval") is not None]
    avg_retrieval_latency = mean(retrieval_latencies) if retrieval_latencies else 0.0

    grounding_failures_total = sum(r.grounding_failures_count for r in results)

    summary = {
        "legal_hit_rate": legal_hit_rate,
        "system_hit_rate": system_hit_rate,
        "avg_retrieval_latency_ms": avg_retrieval_latency,
        "grounding_failures_total": grounding_failures_total if mode == "full" else None,
    }
    return summary


def _apply_gates(summary: Dict[str, Any], mode: str, config: Dict[str, Any]) -> bool:
    failures: List[str] = []

    if summary["legal_hit_rate"] < config["min_legal_must_include_hit_rate"]:
        failures.append(
            f"Legal must-include hit rate {summary['legal_hit_rate']:.2f} below threshold {config['min_legal_must_include_hit_rate']:.2f}"
        )
    if summary["system_hit_rate"] < config["min_system_must_include_hit_rate"]:
        failures.append(
            f"System must-include hit rate {summary['system_hit_rate']:.2f} below threshold {config['min_system_must_include_hit_rate']:.2f}"
        )
    if summary["avg_retrieval_latency_ms"] > config["max_avg_retrieval_latency_ms"]:
        failures.append(
            f"Average retrieval latency {summary['avg_retrieval_latency_ms']:.2f}ms exceeds {config['max_avg_retrieval_latency_ms']:.2f}ms"
        )
    if mode == "full":
        if summary.get("grounding_failures_total", 0) > config["max_grounding_failures_total"]:
            failures.append(
                f"Grounding failures {summary.get('grounding_failures_total', 0)} exceed allowed {config['max_grounding_failures_total']}"
            )

    if failures:
        print("\nRegression gates failed:")
        for msg in failures:
            print(f"- {msg}")
        return False
    return True


# =====================
# Reporting utilities
# =====================
def _environment_settings() -> Dict[str, Any]:
    legal_index = get_legal_index_info()
    manifest = legal_index.get("manifest") or {}

    return {
        "reranker_enabled": RERANK_ENABLED,
        "reranker_model": RERANKER_MODEL,
        "reranker_top_n": RERANK_TOP_N,
        "two_pass_enabled": TWO_PASS_ENABLED,
        "legal_top_k": 8,
        "system_top_k": 6,
        "legal_index_version": legal_index.get("version"),
        "legal_chunk_max_chars": manifest.get("max_chars"),
        "legal_chunk_overlap_chars": manifest.get("overlap_chars"),
    }


def _build_report(run_id: str, results: List[QuestionResult], summary: Dict[str, Any], mode: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": mode,
        "settings": _environment_settings(),
        "summary": summary,
        "results": [
            {
                "id": r.question.id,
                "domain": r.domain,
                "question": r.question.question,
                "legal_object": r.legal_object,
                "queries": r.queries,
                "citations": r.citations,
                "status": r.status,
                "metrics": r.metrics,
                "answer": r.answer if mode == "full" else None,
                "grounding_ok": r.grounding_ok,
                "grounding_failures_count": r.grounding_failures_count,
            }
            for r in results
        ],
    }


def _write_report(run_id: str, report: Dict[str, Any]) -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUN_DIR / f"{run_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return out_path


def _print_summary(run_id: str, summary: Dict[str, Any], mode: str, report_path: Path) -> None:
    print(f"Run ID: {run_id}")
    print(f"Mode: {mode}")
    print(f"Report: {report_path}")
    print("Summary:")
    print(f"  Legal must-include hit rate:  {summary['legal_hit_rate']:.2f}")
    print(f"  System must-include hit rate: {summary['system_hit_rate']:.2f}")
    print(f"  Avg retrieval latency (ms):   {summary['avg_retrieval_latency_ms']:.2f}")
    if mode == "full":
        print(f"  Grounding failures total:     {summary.get('grounding_failures_total', 0)}")


# =====================
# Command-line handling
# =====================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation harness")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval", help="Evaluation mode")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/eval_config.json"),
        help="Path to regression gate configuration",
    )
    return parser.parse_args()


# =====================
# Main entrypoint
# =====================
def main() -> int:
    args = parse_args()
    config = _load_config(args.config)

    questions = _load_questions(LEGAL_FILE) + _load_questions(SYSTEM_FILE)
    if not questions:
        print("No evaluation questions found.")
        return 1

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    mode = args.mode

    results: List[QuestionResult] = []
    for item in questions:
        result = _evaluate_question(item, mode)
        results.append(result)

    summary = _summarize(results, mode)
    report = _build_report(run_id, results, summary, mode)
    report_path = _write_report(run_id, report)
    _print_summary(run_id, summary, mode, report_path)

    passed = _apply_gates(summary, mode, config)
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
