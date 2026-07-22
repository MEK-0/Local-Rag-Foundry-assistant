import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import process_chat_query


def compute_metrics(retrieved_sources: list, expected_source, k: int) -> dict:
    """
    Precision@K / Recall@K / MRR computed against a single expected source
    file. Ground truth here is source-file-level (did the correct document
    surface), not chunk-level - a coarser but honest signal given this eval
    set doesn't have per-chunk relevance annotations yet.

    If expected_source is None (a negative-control question with no
    correct source in the corpus), these metrics don't apply - the caller
    should route to evaluate_negative_control() instead.
    """
    if expected_source is None:
        return {"precision_at_k": None, "recall_at_k": None, "mrr": None}

    top_k = retrieved_sources[:k]
    hits = [1 if src == expected_source else 0 for src in top_k]

    precision_at_k = sum(hits) / len(top_k) if top_k else 0.0
    # Recall@K here means "was the expected source found at all in top K" -
    # with a single ground-truth source per question, recall is binary (0 or 1)
    recall_at_k = 1.0 if any(hits) else 0.0

    mrr = 0.0
    for idx, src in enumerate(retrieved_sources):
        if src == expected_source:
            mrr = 1.0 / (idx + 1)
            break

    return {"precision_at_k": precision_at_k, "recall_at_k": recall_at_k, "mrr": mrr}


def evaluate_negative_control(reply: str) -> bool:
    """
    For questions with no correct answer in the corpus (expected_source_file
    is None in the eval set): checks whether the reply honestly declines or
    reports insufficient context, rather than fabricating a plausible-
    sounding but false answer. This is a hallucination check, not a
    retrieval metric.
    """
    reply_lower = reply.lower()
    honest_decline_signals = [
        "no relevant content", "insufficient context", "unable to produce",
        "not found", "cannot answer", "no information", "not contain",
        "does not", "doesn't contain", "not available in",
    ]
    return any(signal in reply_lower for signal in honest_decline_signals)


def run_single_question(question: dict, advanced_mode: bool, k: int = 5) -> dict:
    result = process_chat_query(question["question"], advanced_mode=advanced_mode)
    chunks_matrix = result.get("chunks_matrix", [])
    retrieved_sources = [c.get("source") for c in chunks_matrix]
    reply = result.get("reply", "")

    expected_source = question.get("expected_source_file")

    if expected_source is None:
        # Negative control - success means an honest decline, not a retrieval hit
        return {
            "id": question["id"],
            "question": question["question"],
            "is_negative_control": True,
            "expected_source": None,
            "retrieved_sources": retrieved_sources,
            "precision_at_k": None,
            "recall_at_k": None,
            "mrr": None,
            "keyword_coverage": None,
            "honest_decline": evaluate_negative_control(reply),
            "reply_preview": reply[:200],
        }

    metrics = compute_metrics(retrieved_sources, expected_source, k)

    reply_lower = reply.lower()
    expected_keywords = question.get("expected_keywords", [])
    keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in reply_lower)
    keyword_coverage = keyword_hits / len(expected_keywords) if expected_keywords else None

    return {
        "id": question["id"],
        "question": question["question"],
        "is_negative_control": False,
        "expected_source": expected_source,
        "retrieved_sources": retrieved_sources,
        "precision_at_k": metrics["precision_at_k"],
        "recall_at_k": metrics["recall_at_k"],
        "mrr": metrics["mrr"],
        "keyword_coverage": keyword_coverage,
        "reply_preview": reply[:200],
    }


def summarize(results: list) -> dict:
    """
    Averages retrieval metrics across scored (non-negative-control)
    questions, and separately reports the hallucination-check pass rate
    for negative-control questions. Keeping these separate avoids a
    misleading blended average - a retrieval metric of None shouldn't
    silently become a 0 or get skipped in a way that inflates the average.
    """
    scored = [r for r in results if not r["is_negative_control"]]
    negative_controls = [r for r in results if r["is_negative_control"]]

    n = len(scored)
    honest_declines = sum(1 for r in negative_controls if r.get("honest_decline"))

    return {
        "avg_precision_at_k": round(sum(r["precision_at_k"] for r in scored) / n, 3) if n else 0.0,
        "avg_recall_at_k": round(sum(r["recall_at_k"] for r in scored) / n, 3) if n else 0.0,
        "avg_mrr": round(sum(r["mrr"] for r in scored) / n, 3) if n else 0.0,
        "negative_control_count": len(negative_controls),
        "negative_control_honest": honest_declines,
    }


def run_eval():
    """
    Runs the labeled eval set against both naive (advanced_mode=False,
    single-pass retrieval, no rerank/grade/compress/follow-up-hop) and
    advanced (full pipeline) configurations, and writes a comparison
    report to docs/eval_report.md.

    Includes multi-part, lexical-overlap, and negative-control questions
    specifically to differentiate naive from advanced - a purely simple,
    keyword-distinctive eval set tends to score both configs at 1.0,
    which measures "is the corpus trivially searchable", not "does the
    advanced pipeline help".
    """
    eval_set_path = "docs/eval_set.json"
    if not os.path.exists(eval_set_path):
        print(f"Eval set not found at {eval_set_path}.")
        return

    with open(eval_set_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Running {len(questions)} questions in NAIVE mode (advanced_mode=False)...")
    naive_results = [run_single_question(q, advanced_mode=False) for q in questions]

    print(f"Running {len(questions)} questions in ADVANCED mode (advanced_mode=True)...")
    advanced_results = [run_single_question(q, advanced_mode=True) for q in questions]

    naive_summary = summarize(naive_results)
    advanced_summary = summarize(advanced_results)

    report_lines = [
        "# Evaluation Report",
        "",
        f"Eval set: `docs/eval_set.json` ({len(questions)} questions)",
        "",
        "**Ground truth is source-file-level** (did the correct document surface in the",
        "top-K retrieved chunks), not chunk-level - a coarser but honest signal given",
        "this eval set doesn't have per-chunk relevance annotations yet. Includes",
        "multi-part, lexical-overlap, and negative-control questions specifically",
        "designed to differentiate naive from advanced retrieval.",
        "",
        "## Summary",
        "",
        "| Config | Precision@5 | Recall@5 | MRR |",
        "|---|---|---|---|",
        f"| Naive (single-pass, no rerank/grade/compress) | {naive_summary['avg_precision_at_k']} | {naive_summary['avg_recall_at_k']} | {naive_summary['avg_mrr']} |",
        f"| Advanced (full pipeline) | {advanced_summary['avg_precision_at_k']} | {advanced_summary['avg_recall_at_k']} | {advanced_summary['avg_mrr']} |",
        "",
        f"**Hallucination check:** {naive_summary['negative_control_honest']}/{naive_summary['negative_control_count']} "
        f"negative-control question(s) answered honestly (naive), "
        f"{advanced_summary['negative_control_honest']}/{advanced_summary['negative_control_count']} (advanced).",
        "",
        "## Per-question detail",
        "",
    ]

    for naive_r, adv_r in zip(naive_results, advanced_results):
        report_lines.append(f"### {naive_r['id']}: {naive_r['question']}")

        if naive_r["is_negative_control"]:
            report_lines.append(f"- Type: negative control (no correct source in corpus)")
            report_lines.append(f"- Naive honest decline: {naive_r['honest_decline']}")
            report_lines.append(f"- Advanced honest decline: {adv_r['honest_decline']}")
            report_lines.append(f"- Naive reply preview: {naive_r['reply_preview']!r}")
            report_lines.append(f"- Advanced reply preview: {adv_r['reply_preview']!r}")
        else:
            report_lines.append(f"- Expected source: `{naive_r['expected_source']}`")
            report_lines.append(f"- Naive retrieved: {naive_r['retrieved_sources']} (MRR: {naive_r['mrr']})")
            report_lines.append(f"- Advanced retrieved: {adv_r['retrieved_sources']} (MRR: {adv_r['mrr']})")
            if naive_r["keyword_coverage"] is not None:
                report_lines.append(f"- Naive keyword coverage: {naive_r['keyword_coverage']:.0%}")
                report_lines.append(f"- Advanced keyword coverage: {adv_r['keyword_coverage']:.0%}")
        report_lines.append("")

    report_text = "\n".join(report_lines)

    output_path = "docs/eval_report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nReport written to {output_path}")
    print(
        f"\nNaive:    Precision@5={naive_summary['avg_precision_at_k']}  "
        f"Recall@5={naive_summary['avg_recall_at_k']}  MRR={naive_summary['avg_mrr']}"
    )
    print(
        f"Advanced: Precision@5={advanced_summary['avg_precision_at_k']}  "
        f"Recall@5={advanced_summary['avg_recall_at_k']}  MRR={advanced_summary['avg_mrr']}"
    )
    print(
        f"\nHallucination check: naive {naive_summary['negative_control_honest']}/{naive_summary['negative_control_count']}, "
        f"advanced {advanced_summary['negative_control_honest']}/{advanced_summary['negative_control_count']}"
    )


if __name__ == "__main__":
    run_eval()