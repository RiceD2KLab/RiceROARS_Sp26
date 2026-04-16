"""
generate_report.py — Run the ROAR pipeline on all labelled .docx files and
produce two timestamped output files inside a  reports/  subfolder:

  reports/pipeline_report_YYYYMMDD_HHMMSS.txt   Step-by-step evidence + review flags
  reports/pipeline_report_YYYYMMDD_HHMMSS.csv   Spreadsheet summary

Human labels are read from the folder structure:
  good_roars/  → "good"
  bad_roars/   → "bad"
  (root)       → label supplied via KNOWN_LABELS dict below

Review flags are attached to each file to highlight cases that need human
attention.  See compute_review_flags() for the full list of flag codes.

Usage:
  python generate_report.py
  python generate_report.py --dir d:/449pipeline --threshold 0.5
  python generate_report.py --out d:/449pipeline/reports
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import time
from pathlib import Path
from typing import Optional

# ── Manually label files that live in the root directory ────────────────────
# All ROAR files are now in good_roars/ or bad_roars/ — no overrides needed.
KNOWN_LABELS: dict[str, str] = {}

DEFAULT_THRESHOLD = 0.5

# ── Weight labels for the weighted score display ─────────────────────────────
WEIGHTS = {"plo": 0.25, "methods": 0.30, "results": 0.30, "plan": 0.15}


# ── Review flag computation ───────────────────────────────────────────────────

def compute_review_flags(result, label: str, threshold: float, error: Optional[str]) -> list[tuple[str, str]]:
    """
    Return a list of (FLAG_CODE, explanation) tuples for a pipeline result.

    Flag codes
    ----------
    PARSE_ERROR         JSON parsing failed — output was malformed.
    ALL_ZERO_SCORES     Every section scored 0. Possible extraction failure or
                        overly strict model; check the extracted sections.
    VERIFIER_INVERTED   Verifier flipped ALL scores (all 1→0 or 0→1).
                        Strong sign of a prompt misunderstanding.
    INCONSISTENT        Models did not agree after max feedback iterations.
                        Human should decide the final score.
    FEEDBACK_REQUIRED   One or more feedback loops were triggered (iterations>0).
                        Scores may be less stable.
    BORDERLINE_SCORE    Weighted score is in the 40–74% ambiguous range.
                        Pipeline prediction may be unreliable near threshold.
    PLO_ONLY_FAIL       Only PLO scored 0; all other sections passed.
                        LLaMA is often over-strict on PLO wording.
    PLAN_ONLY_FAIL      Only Plan scored 0; all other sections passed.
                        "No changes needed" is sometimes incorrectly penalised.
    LABEL_MISMATCH      Pipeline prediction does not match human label.
                        Definite review needed.
    """
    flags: list[tuple[str, str]] = []

    if error:
        flags.append(("PARSE_ERROR",
                       f"Pipeline failed with error: {error[:120]}"))
        return flags

    if result is None:
        return [("PARSE_ERROR", "No result object returned.")]

    score  = result.weighted_score
    pred   = "good" if score >= threshold else "bad"
    scores = result.final_scores

    # All sections zero
    if all(getattr(scores, f) == 0 for f in ("plo", "methods", "results", "plan")):
        flags.append(("ALL_ZERO_SCORES",
                       "Every section scored 0. Check extracted text — sections may be empty "
                       "or the model may have been over-strict."))

    # Verifier inverted all scores
    for step in result.audit_trail:
        if "Verification" in step.step_name and step.differences:
            all_flipped = all(
                d.llama_score != d.correct_score for d in step.differences
            ) and len(step.differences) == 4
            if all_flipped:
                flags.append(("VERIFIER_INVERTED",
                               f"Verifier ({step.step_name}) flipped ALL 4 scores. "
                               "This is usually a prompt misinterpretation, not a real disagreement."))
                break

    # Did not converge
    if not result.consistent:
        flags.append(("INCONSISTENT",
                       f"Models did not agree after {result.iterations} feedback iteration(s). "
                       "Final scores use LLaMA's last revision — human review recommended."))

    # Feedback was needed
    elif result.iterations > 0:
        flags.append(("FEEDBACK_REQUIRED",
                       f"{result.iterations} feedback iteration(s) were needed before convergence. "
                       "Scores are less stable than zero-iteration results."))

    # Borderline score
    if 0.40 <= score < 0.75:
        flags.append(("BORDERLINE_SCORE",
                       f"Score {score*100:.0f}% is in the ambiguous zone (40–74%). "
                       "Pipeline prediction may be unreliable. Human review recommended."))

    # PLO only fail
    if (scores.plo == 0 and scores.methods == 1
            and scores.results == 1 and scores.plan == 1):
        flags.append(("PLO_ONLY_FAIL",
                       "Only PLO failed. LLaMA is often over-strict about 'measurability' — "
                       "broad learner outcomes are valid per Rice SACSCOC guidelines."))

    # Plan only fail
    if (scores.plan == 0 and scores.plo == 1
            and scores.methods == 1 and scores.results == 1):
        flags.append(("PLAN_ONLY_FAIL",
                       "Only Plan failed. Check whether the plan says 'no changes needed + continue X' — "
                       "this should count as actionable."))

    # Prediction vs label
    if label != "unknown" and pred != label:
        flags.append(("LABEL_MISMATCH",
                       f"Pipeline predicted {pred.upper()} but human labeled it {label.upper()}. "
                       "This file requires human review."))

    return flags if flags else [("NONE", "No issues detected. Result appears reliable.")]


# ── File discovery ────────────────────────────────────────────────────────────

def find_files(root: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*.docx")):
        parent = path.parent.name.lower()
        if "good" in parent:
            label = "good"
        elif "bad" in parent:
            label = "bad"
        else:
            label = KNOWN_LABELS.get(path.name, "unknown")
        pairs.append((path, label))
    return pairs


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(path: Path):
    from pipeline.extractor import SectionExtractor
    from pipeline.roar_pipeline import ROARPipeline

    pipeline = ROARPipeline()
    sections = SectionExtractor.extract_from_docx(path)
    result   = pipeline.run(pre_sections=sections)
    return result


# ── Formatting helpers ────────────────────────────────────────────────────────

def score_bar(score: int | None) -> str:
    if score is None: return "?"
    return "PASS (1)" if score == 1 else "FAIL (0)"


def fmt_scores(s) -> str:
    return (f"PLO={s.plo}  Methods={s.methods}  Results={s.results}  Plan={s.plan}"
            f"  →  Weighted={(s.plo*0.25 + s.methods*0.30 + s.results*0.30 + s.plan*0.15)*100:.0f}%")


def trunc(text: str, n: int = 300) -> str:
    text = text.strip().replace("\n", " ")
    return text[:n] + ("…" if len(text) > n else "")


# ── TXT report builder ────────────────────────────────────────────────────────

def build_txt(rows: list[dict], threshold: float) -> str:
    buf = io.StringIO()
    w   = buf.write

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    border = "=" * 90

    w(f"{border}\n")
    w(f"  ROAR QUALITY ASSESSMENT — PIPELINE EVALUATION REPORT\n")
    w(f"  Generated : {now}\n")
    import config as _cfg
    _ev_label = f"{_cfg.EVALUATOR_MODEL}  (backend: {_cfg.EVALUATOR_BACKEND})"
    _ve_label = f"{_cfg.VERIFIER_AZURE_DEPLOYMENT}  (backend: {_cfg.VERIFIER_BACKEND})"
    w(f"  Evaluator : {_ev_label}\n")
    w(f"  Verifier  : {_ve_label}\n")
    w(f"  Threshold : score >= {threshold*100:.0f}% → GOOD  |  score < {threshold*100:.0f}% → BAD\n")
    w(f"{border}\n\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    w("SUMMARY\n")
    w("-" * 90 + "\n")
    header = f"  {'#':>2}  {'FILE':<44} {'LABEL':>7} {'SCORE':>6}  {'PRED':>4}  {'MATCH':>5}\n"
    w(header)
    w("-" * 90 + "\n")

    correct = scored = 0
    for r in rows:
        label = r["human_label"]
        if r["error"]:
            w(f"  {r['idx']:>2}  {r['file']:<44}  ERROR\n")
            continue
        score = r["result"].weighted_score
        pred  = "good" if score >= threshold else "bad"
        match = "  OK" if (label != "unknown" and pred == label) else ("MISS" if label != "unknown" else "  --")
        if label != "unknown":
            scored += 1
            if pred == label: correct += 1
        w(f"  {r['idx']:>2}  {r['file']:<44} {label:>7}  {score*100:5.1f}%  {pred:>4}  {match}\n")

    w("-" * 90 + "\n")
    if scored:
        w(f"  Accuracy: {correct}/{scored} = {correct/scored*100:.0f}%"
          f"  |  Avg score: {sum(r['result'].weighted_score for r in rows if not r['error'])/len([r for r in rows if not r['error']])*100:.1f}%\n")
    w("\n\n")

    # ── Per-file detailed section ─────────────────────────────────────────────
    w("DETAILED RESULTS\n")

    for r in rows:
        w("=" * 90 + "\n")
        label  = r["human_label"]
        result = r["result"]
        score  = result.weighted_score if result else None
        pred   = ("good" if score >= threshold else "bad") if score is not None else "ERROR"
        match  = ("OK — matches human label" if pred == label
                  else ("MISS — does NOT match human label" if label != "unknown"
                  else "human label: unknown"))

        w(f"FILE    : {r['file']}\n")
        w(f"FOLDER  : {r['folder']}\n")
        w(f"LABEL   : {label}\n")
        w(f"ELAPSED : {r['elapsed_sec']}s\n")

        if r["error"]:
            w(f"\n  ERROR: {r['error']}\n\n")
            continue

        w("\n")

        # ── Step 1: Extracted sections ────────────────────────────────────────
        w("STEP 1 — SECTION EXTRACTION\n")
        s = result.sections
        if s:
            w(f"  PLO     : {trunc(s.plo, 250)}\n")
            w(f"  Methods : {trunc(s.methods, 250)}\n")
            w(f"  Results : {trunc(s.results, 250)}\n")
            w(f"  Plan    : {trunc(s.plan, 250)}\n")
        w("\n")

        # ── Step 2+ : Audit trail ─────────────────────────────────────────────
        for step in result.audit_trail:
            w(f"{step.step_name.upper()}\n")
            w(f"  Scores  : {fmt_scores(step.scores)}\n")

            if step.reasoning:
                w(f"  Reasoning:\n")
                for field in ("plo", "methods", "results", "plan"):
                    text = getattr(step.reasoning, field, "")
                    if text:
                        w(f"    [{field.upper():8}] {trunc(text, 200)}\n")

            if step.consistent is not None:
                w(f"  Consistent : {'YES — pipeline converged' if step.consistent else 'NO — disagreement found'}\n")
                if step.differences:
                    w(f"  Differences ({len(step.differences)}):\n")
                    for d in step.differences:
                        w(f"    • {d.field}: LLaMA={d.llama_score} → Verifier={d.correct_score}  \"{trunc(d.reason, 120)}\"\n")

            if step.changes:
                w(f"  Changes made ({len(step.changes)}):\n")
                for c in step.changes:
                    w(f"    • {c.field}: {c.old_score} → {c.new_score}  \"{trunc(c.reason, 120)}\"\n")

            w("\n")

        # ── Final verdict ─────────────────────────────────────────────────────
        w("FINAL VERDICT\n")
        w(f"  PLO     : {score_bar(result.final_scores.plo)}\n")
        w(f"  Methods : {score_bar(result.final_scores.methods)}\n")
        w(f"  Results : {score_bar(result.final_scores.results)}\n")
        w(f"  Plan    : {score_bar(result.final_scores.plan)}\n")
        w(f"  Weighted Score    : {result.weighted_score*100:.1f}%\n")
        w(f"  Pipeline Pred     : {pred.upper()}\n")
        w(f"  Human Label       : {label.upper()}\n")
        w(f"  Match             : {match}\n")
        w(f"  Consistent        : {'Yes' if result.consistent else 'No'}\n")
        w(f"  Feedback Iter     : {result.iterations}\n")

        # ── Review flags ──────────────────────────────────────────────────────
        flags = compute_review_flags(result, label, threshold, r["error"])
        w("\nREVIEW FLAGS\n")
        for code, explanation in flags:
            marker = "!! " if code not in ("NONE",) else "   "
            w(f"  {marker}[{code}]\n     {explanation}\n")
        w("\n")

    w("=" * 90 + "\n")
    w("END OF REPORT\n")
    return buf.getvalue()


# ── CSV builder ───────────────────────────────────────────────────────────────

def build_csv(rows: list[dict], threshold: float) -> str:
    buf = io.StringIO()
    fieldnames = [
        "file", "folder", "human_label",
        "plo_score", "methods_score", "results_score", "plan_score",
        "weighted_score_pct", "pipeline_pred", "match",
        "consistent", "feedback_iterations", "elapsed_sec",
        "initial_plo", "initial_methods", "initial_results", "initial_plan",
        "reasoning_plo", "reasoning_methods", "reasoning_results", "reasoning_plan",
        "verifier_consistent_iter1", "verifier_differences_iter1",
        "review_flags", "review_details",
        "error",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()

    for r in rows:
        label  = r["human_label"]
        result = r["result"]

        row: dict = {
            "file":    r["file"],
            "folder":  r["folder"],
            "human_label": label,
            "elapsed_sec": r["elapsed_sec"],
            "error":   r["error"] or "",
        }

        if result and not r["error"]:
            score = result.weighted_score
            pred  = "good" if score >= threshold else "bad"
            match = ("OK" if (label != "unknown" and pred == label)
                     else ("MISS" if label != "unknown" else "--"))
            row.update({
                "plo_score":      result.final_scores.plo,
                "methods_score":  result.final_scores.methods,
                "results_score":  result.final_scores.results,
                "plan_score":     result.final_scores.plan,
                "weighted_score_pct": round(score * 100, 1),
                "pipeline_pred":  pred,
                "match":          match,
                "consistent":     result.consistent,
                "feedback_iterations": result.iterations,
            })
            # Reasoning from final scores
            if result.reasoning:
                row["reasoning_plo"]     = trunc(result.reasoning.plo, 200)
                row["reasoning_methods"] = trunc(result.reasoning.methods, 200)
                row["reasoning_results"] = trunc(result.reasoning.results, 200)
                row["reasoning_plan"]    = trunc(result.reasoning.plan, 200)
            # Initial scores (Step 2)
            initial = next((s for s in result.audit_trail if "Initial" in s.step_name), None)
            if initial:
                row["initial_plo"]     = initial.scores.plo
                row["initial_methods"] = initial.scores.methods
                row["initial_results"] = initial.scores.results
                row["initial_plan"]    = initial.scores.plan
            # Verifier result (Step 3, iteration 1)
            v1 = next((s for s in result.audit_trail if "Verification" in s.step_name
                       and "iteration 1" in s.step_name), None)
            if v1:
                row["verifier_consistent_iter1"]  = v1.consistent
                row["verifier_differences_iter1"] = (
                    "; ".join(f"{d.field}:{d.llama_score}→{d.correct_score}" for d in v1.differences)
                    if v1.differences else ""
                )
            # Review flags
            flags = compute_review_flags(result, label, threshold, r["error"])
            row["review_flags"]   = "|".join(code for code, _ in flags)
            row["review_details"] = " | ".join(f"[{code}] {expl}" for code, expl in flags)
        else:
            for f in fieldnames:
                if f not in row:
                    row[f] = ""

        writer.writerow(row)

    return buf.getvalue()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",       default=".", metavar="PATH")
    parser.add_argument("--threshold", type=float,  default=DEFAULT_THRESHOLD)
    parser.add_argument("--out",       default=".",  metavar="DIR",
                        help="Output directory for report files (default: current dir)")
    args = parser.parse_args()

    root      = Path(args.dir).resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Always save inside a  reports/  subfolder under the pipeline root
    base_out = Path(args.out).resolve()
    outdir   = base_out / "reports"
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = find_files(root)

    if not pairs:
        print(f"No .docx files found under {root}")
        return

    print(f"\nRunning pipeline on {len(pairs)} ROAR file(s)…")
    import config as _cfg
    print(f"Evaluator : {_cfg.EVALUATOR_MODEL}  (backend: {_cfg.EVALUATOR_BACKEND})")
    print(f"Verifier  : {_cfg.VERIFIER_AZURE_DEPLOYMENT}  (backend: {_cfg.VERIFIER_BACKEND})\n")

    rows: list[dict] = []
    for i, (path, label) in enumerate(pairs, 1):
        print(f"  [{i:02}/{len(pairs):02}] [{label:>7}]  {path.name} ...", end="", flush=True)
        start = time.time()
        error = None
        result = None
        try:
            result = run_pipeline(path)
            pred = "good" if result.weighted_score >= args.threshold else "bad"
            match = ("OK" if (label != "unknown" and pred == label)
                     else ("MISS" if label != "unknown" else "--"))
            print(f"  {result.weighted_score*100:.0f}%  pred={pred} [{match}]  ({time.time()-start:.1f}s)")
        except Exception as exc:
            error = str(exc)
            print(f"  ERROR: {error[:60]}")

        rows.append({
            "idx":        i,
            "file":       path.name,
            "folder":     path.parent.name,
            "human_label": label,
            "elapsed_sec": round(time.time() - start, 1),
            "result":     result,
            "error":      error,
        })

    # ── Write outputs (timestamped so multiple runs never overwrite each other) ─
    txt_path = outdir / f"pipeline_report_{timestamp}.txt"
    csv_path = outdir / f"pipeline_report_{timestamp}.csv"

    txt_path.write_text(build_txt(rows, args.threshold), encoding="utf-8")
    csv_path.write_text(build_csv(rows, args.threshold), encoding="utf-8")

    # ── Final summary ─────────────────────────────────────────────────────────
    successful = [r for r in rows if not r["error"]]
    labeled    = [r for r in successful if r["human_label"] != "unknown"]
    correct    = sum(1 for r in labeled
                     if ("good" if r["result"].weighted_score >= args.threshold else "bad") == r["human_label"])

    print(f"\n  Files processed : {len(rows)}")
    print(f"  Errors          : {len(rows) - len(successful)}")
    if labeled:
        print(f"  Accuracy        : {correct}/{len(labeled)} = {correct/len(labeled)*100:.0f}%")
    print(f"\n  Reports saved to:")
    print(f"    {txt_path}")
    print(f"    {csv_path}\n")


if __name__ == "__main__":
    main()
