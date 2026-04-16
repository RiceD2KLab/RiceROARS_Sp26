"""
evaluation_test_synthetic.py — Run the ROAR pipeline on synthetic CSV data.

Reads rows from a CSV where each row is a ROAR with pre-split section text
and ground-truth labels (both overall and per-section).  Runs a specified
model set × all prompt sets (A/B/C) with parallel batching.

Outputs:
  evaluation_results/synth_eval_YYYYMMDD_HHMMSS.txt    Full report
  evaluation_results/synth_eval_YYYYMMDD_HHMMSS.csv    Per-row scores
  evaluation_results/synth_eval_YYYYMMDD_HHMMSS.json   Structured verdicts

Usage:
  python evaluation_test_synthetic.py                          # first 100 rows, 6 workers
  python evaluation_test_synthetic.py --rows 200 --workers 10
  python evaluation_test_synthetic.py --csv path/to/file.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from models.schemas import PipelineResult, ROARSections, SectionScores

# ── Column mapping: CSV column → ROARSections field ──────────────────────────

CSV_TO_SECTION = {
    "PLO_Description":  "plo",
    "Assessment_Method": "methods",
    "ROAR":             "results",
    "ROAR_Follow_up":   "plan",
}

CSV_SECTION_LABEL = {
    "PLO_Description_label":  "plo",
    "Assessment_Method_label": "methods",
    "ROAR_label":             "results",
    "ROAR_Follow_up_label":   "plan",
}

DEFAULT_ROWS    = 100
DEFAULT_WORKERS = 6


# ── CSV reader ───────────────────────────────────────────────────────────────

def load_csv_rows(csv_path: Path, max_rows: int) -> list[dict]:
    """Read up to max_rows from the CSV and return normalised dicts."""
    rows: list[dict] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader):
            if i >= max_rows:
                break

            sections = ROARSections(
                plo=raw.get("PLO_Description", "").strip(),
                methods=raw.get("Assessment_Method", "").strip(),
                results=raw.get("ROAR", "").strip(),
                plan=raw.get("ROAR_Follow_up", "").strip(),
            )

            overall_label = "good" if raw.get("label", "0").strip() == "1" else "bad"

            section_labels: dict[str, int] = {}
            for csv_col, sec_name in CSV_SECTION_LABEL.items():
                val = raw.get(csv_col, "").strip()
                section_labels[sec_name] = int(val) if val in ("0", "1") else -1

            rows.append({
                "row_idx": i,
                "program": raw.get("Program", ""),
                "plo":     raw.get("PLO", ""),
                "sections": sections,
                "human_label": overall_label,
                "section_labels": section_labels,
            })
    return rows


# ── Pipeline runner ──────────────────────────────────────────────────────────

def build_pipeline(model_set: dict, prompt_set: dict):
    from pipeline.roar_pipeline import ROARPipeline

    ev_cfg = model_set["evaluator"]
    ve_cfg = model_set["verifier"]
    return ROARPipeline(
        evaluator_model=ev_cfg["model"],
        verifier_model=ve_cfg["model"],
        evaluator_backend=ev_cfg["backend"],
        verifier_backend=ve_cfg["backend"],
        prompt_set=prompt_set,
    )


def run_one(sections: ROARSections, model_set: dict, prompt_set: dict) -> tuple:
    """Return (PipelineResult | None, error_str | None, elapsed_sec)."""
    pipeline = build_pipeline(model_set, prompt_set)
    start = time.time()
    try:
        result = pipeline.run(pre_sections=sections)
        return result, None, round(time.time() - start, 1)
    except Exception as exc:
        return None, str(exc)[:200], round(time.time() - start, 1)


# ── Classification strategies ────────────────────────────────────────────────

def classify_weighted(result: PipelineResult, threshold: float = 0.5) -> str:
    return "good" if result.weighted_score >= threshold else "bad"


def classify_strict(result: PipelineResult) -> str:
    s = result.final_scores
    return "good" if (s.plo == 1 and s.methods == 1 and s.results == 1 and s.plan == 1) else "bad"


def failed_sections(scores: SectionScores) -> list[str]:
    return [sec for sec in ("plo", "methods", "results", "plan")
            if getattr(scores, sec) == 0]


def build_verdict(result: PipelineResult, human_label: str) -> dict:
    s = result.final_scores
    s_pred = classify_strict(result)
    fails  = failed_sections(s)

    section_reasons: dict[str, str] = {}
    for sec in ("plo", "methods", "results", "plan"):
        txt = ""
        if result.reasoning:
            txt = getattr(result.reasoning, sec, "") or ""
        section_reasons[sec] = txt.strip()

    if fails:
        parts = []
        for sec in fails:
            r = section_reasons.get(sec, "")
            short = r[:300] if r else "no reasoning provided"
            parts.append(f"{sec.upper()}: {short}")
        reason = " | ".join(parts)
    else:
        reason = ""

    return {
        "prediction": s_pred,
        "passed": s_pred == "good",
        "weighted_score": result.weighted_score,
        "scores": {"plo": s.plo, "methods": s.methods, "results": s.results, "plan": s.plan},
        "failed_sections": fails,
        "reason": reason,
        "section_reasons": section_reasons,
        "consistent": result.consistent,
        "iterations": result.iterations,
    }


# ── Metrics ──────────────────────────────────────────────────────────────────

STRATEGIES = ["weighted", "strict"]


def _classify(result, strat):
    return classify_weighted(result) if strat == "weighted" else classify_strict(result)


def compute_overall_metrics(rows: list[dict], set_key: str, strat: str) -> dict:
    tp = tn = fp = fn_ = 0
    for r in rows:
        result = r["results"].get(set_key, {}).get("result")
        if result is None:
            continue
        label = r["human_label"]
        pred  = _classify(result, strat)
        if   label == "good" and pred == "good": tp  += 1
        elif label == "bad"  and pred == "bad":  tn  += 1
        elif label == "bad"  and pred == "good": fp  += 1
        elif label == "good" and pred == "bad":  fn_ += 1

    total = tp + tn + fp + fn_
    acc  = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn_) if (tp + fn_) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn_, "total": total,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def compute_section_accuracy(rows: list[dict], set_key: str) -> dict[str, dict]:
    """Compare predicted per-section scores against ground-truth section labels."""
    stats: dict[str, dict] = {}
    for sec in ("plo", "methods", "results", "plan"):
        correct = total = 0
        tp = tn = fp = fn_ = 0
        for r in rows:
            gt = r["section_labels"].get(sec, -1)
            if gt == -1:
                continue
            result = r["results"].get(set_key, {}).get("result")
            if result is None:
                continue
            pred = getattr(result.final_scores, sec)
            total += 1
            if pred == gt:
                correct += 1
            if gt == 1 and pred == 1: tp += 1
            elif gt == 0 and pred == 0: tn += 1
            elif gt == 0 and pred == 1: fp += 1
            elif gt == 1 and pred == 0: fn_ += 1

        acc  = correct / total if total else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec  = tp / (tp + fn_) if (tp + fn_) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        stats[sec] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                      "tp": tp, "tn": tn, "fp": fp, "fn": fn_, "total": total}
    return stats


def compute_avg_runtime(rows: list[dict], set_keys: list[str]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for k in set_keys:
        times = []
        for r in rows:
            res = r["results"].get(k, {})
            if res.get("result") is not None and res.get("error") is None:
                times.append(res["elapsed"])
        if times:
            stats[k] = {"mean": round(sum(times) / len(times), 1),
                        "min": min(times), "max": max(times),
                        "total": round(sum(times), 1), "count": len(times)}
        else:
            stats[k] = {"mean": 0, "min": 0, "max": 0, "total": 0, "count": 0}
    return stats


# ── ASCII helpers ────────────────────────────────────────────────────────────

def confusion_matrix_str(m: dict) -> str:
    tp, tn, fp, fn_ = m["tp"], m["tn"], m["fp"], m["fn"]
    return "\n".join([
        "                  PREDICTED",
        "                  good    bad",
        f"  ACTUAL  good  [ {tp:3d}  | {fn_:3d} ]  ← TP | FN",
        f"           bad  [ {fp:3d}  | {tn:3d} ]  ← FP | TN",
    ])


def trunc(t: str, n: int = 200) -> str:
    t = t.strip().replace("\n", " ")
    return t[:n] + ("…" if len(t) > n else "")


# ── Report builders ─────────────────────────────────────────────────────────

def build_txt(rows: list[dict], set_keys: list[str], set_labels: dict[str, str],
              model_label: str, overall_metrics: dict, section_metrics: dict,
              runtime_stats: dict) -> str:
    buf = io.StringIO()
    w   = buf.write
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    w("=" * 110 + "\n")
    w("  ROAR SYNTHETIC DATA EVALUATION\n")
    w(f"  Generated : {now}\n")
    w(f"  Model Set : {model_label}\n")
    w(f"  Rows      : {len(rows)}\n")
    w("=" * 110 + "\n\n")

    # ── Prompt sets ──────────────────────────────────────────────────────────
    w("PROMPT SETS TESTED\n")
    w("-" * 60 + "\n")
    for k in set_keys:
        from prompt_sets import PROMPT_SETS
        w(f"  {k}\n    {PROMPT_SETS[k]['description']}\n")
    w("\n")

    # ── Runtime ──────────────────────────────────────────────────────────────
    w("AVERAGE RUNTIME PER PROMPT SET\n")
    w("─" * 90 + "\n")
    w(f"  {'Prompt Set':<32}  {'Avg':>8}  {'Min':>8}  {'Max':>8}  {'Total':>9}  {'Runs':>5}\n")
    w("─" * 90 + "\n")
    for k in set_keys:
        rs = runtime_stats[k]
        w(f"  {k:<32}  {rs['mean']:7.1f}s  {rs['min']:7.1f}s  {rs['max']:7.1f}s  {rs['total']:8.1f}s  {rs['count']:5d}\n")
    w("\n")

    # ── Overall metrics per strategy ─────────────────────────────────────────
    for strat in STRATEGIES:
        strat_label = "WEIGHTED (≥50%)" if strat == "weighted" else "STRICT (all-must-pass)"
        w(f"{'─' * 110}\n")
        w(f"OVERALL METRICS — {strat_label}\n")
        w(f"{'─' * 110}\n")
        w(f"  {'Metric':<22}")
        for k in set_keys:
            w(f"  {k:<30}")
        w("\n" + "─" * 110 + "\n")
        for metric, label in [("accuracy", "Accuracy"), ("precision", "Precision"),
                               ("recall", "Recall"), ("f1", "F1-Score")]:
            w(f"  {label:<22}")
            for k in set_keys:
                v = overall_metrics[strat][k][metric]
                w(f"  {v*100:5.1f}%{'':<24}")
            w("\n")
        w("─" * 110 + "\n")
        w(f"  {'TP / TN / FP / FN':<22}")
        for k in set_keys:
            m = overall_metrics[strat][k]
            w(f"  {m['tp']}TP {m['tn']}TN {m['fp']}FP {m['fn']}FN{'':<17}")
        w("\n\n")

    # ── Strategy delta ───────────────────────────────────────────────────────
    w("=" * 110 + "\n")
    w("STRATEGY DELTA  (strict − weighted)\n")
    w("=" * 110 + "\n")
    w(f"  {'Prompt Set':<30}  {'ΔAcc':>7}  {'ΔPrec':>7}  {'ΔRec':>7}  {'ΔF1':>7}  {'ΔFP':>5}  {'ΔFN':>5}\n")
    w("-" * 90 + "\n")
    for k in set_keys:
        mw = overall_metrics["weighted"][k]
        ms = overall_metrics["strict"][k]
        w(f"  {k:<30}  {(ms['accuracy']-mw['accuracy'])*100:+6.1f}%"
          f"  {(ms['precision']-mw['precision'])*100:+6.1f}%"
          f"  {(ms['recall']-mw['recall'])*100:+6.1f}%"
          f"  {(ms['f1']-mw['f1'])*100:+6.1f}%"
          f"  {ms['fp']-mw['fp']:+4d}   {ms['fn']-mw['fn']:+4d}\n")
    w("\n")

    # ── Confusion matrices ───────────────────────────────────────────────────
    for strat in STRATEGIES:
        strat_label = "WEIGHTED" if strat == "weighted" else "STRICT"
        w(f"CONFUSION MATRICES — {strat_label}\n")
        for k in set_keys:
            w(f"\n  {set_labels[k]}\n")
            for line in confusion_matrix_str(overall_metrics[strat][k]).splitlines():
                w(f"  {line}\n")
        w("\n")

    # ── Per-section accuracy ─────────────────────────────────────────────────
    w("=" * 110 + "\n")
    w("PER-SECTION ACCURACY  (predicted section score vs ground-truth section label)\n")
    w("=" * 110 + "\n\n")
    for k in set_keys:
        w(f"  [{set_labels[k]}]\n")
        w(f"  {'Section':<10}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}\n")
        w("  " + "─" * 70 + "\n")
        for sec in ("plo", "methods", "results", "plan"):
            sm = section_metrics[k][sec]
            w(f"  {sec.upper():<10}  {sm['accuracy']*100:6.1f}%  {sm['precision']*100:6.1f}%"
              f"  {sm['recall']*100:6.1f}%  {sm['f1']*100:6.1f}%"
              f"  {sm['tp']:4d} {sm['tn']:4d} {sm['fp']:4d} {sm['fn']:4d}\n")
        w("\n")

    # ── Per-file summary table ───────────────────────────────────────────────
    w("=" * 110 + "\n")
    w("PER-ROW RESULTS (first 200 rows shown)\n")
    w("=" * 110 + "\n")
    for k in set_keys:
        short = k.split("_")[0]
        w(f"\n  [{set_labels[k]}]\n")
        w(f"  {'#':>4}  {'PROGRAM':<30}  {'LABEL':>6}  {'SCORE':>6}  {'W':>5}  {'S':>5}  {'FAILED':<30}  {'OK':>4}\n")
        w("  " + "─" * 100 + "\n")
        for r in rows[:200]:
            res = r["results"].get(k, {})
            result = res.get("result")
            err    = res.get("error")
            idx    = r["row_idx"]
            prog   = r["program"][:28]
            label  = r["human_label"]

            if err or result is None:
                w(f"  {idx:4d}  {prog:<30}  {label:>6}  {'ERR':>6}  {'':>5}  {'':>5}  {'':<30}  {'':<4}\n")
                continue

            wp = classify_weighted(result)
            sp = classify_strict(result)
            fails = failed_sections(result.final_scores)
            fails_str = ",".join(s.upper() for s in fails) if fails else "—"
            ok = "OK" if sp == label else "MISS"
            w(f"  {idx:4d}  {prog:<30}  {label:>6}  {result.weighted_score*100:5.0f}%"
              f"  {wp:>5}  {sp:>5}  {fails_str:<30}  {ok:>4}\n")
        w("\n")

    w("=" * 110 + "\n")
    w("END OF REPORT\n")
    return buf.getvalue()


def build_csv(rows: list[dict], set_keys: list[str]) -> str:
    buf = io.StringIO()
    base = ["row_idx", "program", "plo_id", "human_label",
            "gt_plo", "gt_methods", "gt_results", "gt_plan"]
    score_fields = []
    for k in set_keys:
        short = k.split("_")[0]
        score_fields += [
            f"{short}_plo", f"{short}_methods", f"{short}_results", f"{short}_plan",
            f"{short}_weighted_pct",
            f"{short}_weighted_pred", f"{short}_strict_pred",
            f"{short}_strict_match",
            f"{short}_failed_sections", f"{short}_reason",
            f"{short}_elapsed_sec", f"{short}_error",
        ]
    writer = csv.DictWriter(buf, fieldnames=base + score_fields, lineterminator="\n")
    writer.writeheader()

    for r in rows:
        row = {
            "row_idx": r["row_idx"], "program": r["program"], "plo_id": r["plo"],
            "human_label": r["human_label"],
            "gt_plo": r["section_labels"]["plo"],
            "gt_methods": r["section_labels"]["methods"],
            "gt_results": r["section_labels"]["results"],
            "gt_plan": r["section_labels"]["plan"],
        }
        for k in set_keys:
            short = k.split("_")[0]
            res = r["results"].get(k, {})
            result = res.get("result")
            err    = res.get("error", "")
            elapsed = res.get("elapsed", 0)
            if result and not err:
                v = build_verdict(result, r["human_label"])
                s = result.final_scores
                wp = classify_weighted(result)
                label = r["human_label"]
                s_ok = "OK" if v["prediction"] == label else "MISS"
                row.update({
                    f"{short}_plo": s.plo, f"{short}_methods": s.methods,
                    f"{short}_results": s.results, f"{short}_plan": s.plan,
                    f"{short}_weighted_pct": round(result.weighted_score * 100, 1),
                    f"{short}_weighted_pred": wp,
                    f"{short}_strict_pred": v["prediction"],
                    f"{short}_strict_match": s_ok,
                    f"{short}_failed_sections": "|".join(v["failed_sections"]),
                    f"{short}_reason": v["reason"],
                    f"{short}_elapsed_sec": elapsed,
                    f"{short}_error": "",
                })
            else:
                for f in [f"{short}_plo", f"{short}_methods", f"{short}_results",
                          f"{short}_plan", f"{short}_weighted_pct",
                          f"{short}_weighted_pred", f"{short}_strict_pred",
                          f"{short}_strict_match", f"{short}_failed_sections",
                          f"{short}_reason", f"{short}_elapsed_sec"]:
                    row[f] = ""
                row[f"{short}_error"] = err or "no result"
        writer.writerow(row)
    return buf.getvalue()


def build_json(rows: list[dict], set_keys: list[str],
               runtime_stats: dict, model_label: str) -> str:
    output = []
    for r in rows:
        entry = {
            "row_idx": r["row_idx"],
            "program": r["program"],
            "plo": r["plo"],
            "human_label": r["human_label"],
            "section_labels": r["section_labels"],
            "verdicts": {},
        }
        for k in set_keys:
            res = r["results"].get(k, {})
            result = res.get("result")
            err    = res.get("error")
            elapsed = res.get("elapsed", 0)
            if err or result is None:
                entry["verdicts"][k] = {"error": err or "no result", "elapsed_sec": elapsed}
            else:
                v = build_verdict(result, r["human_label"])
                v["elapsed_sec"] = elapsed
                entry["verdicts"][k] = v
        output.append(entry)

    wrapper = {
        "generated": datetime.datetime.now().isoformat(),
        "model_set": model_label,
        "strategy": "strict_all_must_pass",
        "runtime_stats": runtime_stats,
        "rows": output,
    }
    return json.dumps(wrapper, indent=2, ensure_ascii=False)


# ── Parallel runner ──────────────────────────────────────────────────────────

_print_lock = threading.Lock()

def _safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def _run_one_job(job: dict) -> dict:
    row      = job["row"]
    set_key  = job["set_key"]
    ps       = job["prompt_set"]
    ms       = job["model_set"]
    job_idx  = job["job_idx"]
    n_total  = job["n_total"]

    _safe_print(f"  [{job_idx:4d}/{n_total}]  row {row['row_idx']:4d}  ×  {set_key} ...", flush=True)
    result, error, elapsed = run_one(row["sections"], ms, ps)

    if error:
        _safe_print(f"  [{job_idx:4d}/{n_total}]  row {row['row_idx']:4d}  ×  {set_key}  ERROR ({elapsed}s)")
    else:
        sp = classify_strict(result)
        ok = "OK" if sp == row["human_label"] else "MISS"
        _safe_print(
            f"  [{job_idx:4d}/{n_total}]  row {row['row_idx']:4d}  ×  {set_key}"
            f"  {result.weighted_score*100:.0f}%  s={sp}  [{ok}]  ({elapsed}s)")

    return {
        "row_idx": row["row_idx"],
        "set_key": set_key,
        "result": result, "error": error, "elapsed": elapsed,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run ROAR pipeline on synthetic CSV data.")
    parser.add_argument("--csv", default="synthetic_data/SYNTHETIC_DATA_labeled_split_with_section_labels.csv",
                        metavar="PATH", help="Path to the synthetic CSV file.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                        help=f"Number of rows to evaluate (default {DEFAULT_ROWS}).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, metavar="N",
                        help=f"Parallel threads (default {DEFAULT_WORKERS}).")
    parser.add_argument("--model-set", default="MS3_DeepSeek_o4mini",
                        help="Model set key from model_sets.py (default MS3_DeepSeek_o4mini).")
    parser.add_argument("--dir", default=".", metavar="PATH",
                        help="Project root for output directory.")
    args = parser.parse_args()

    from prompt_sets import PROMPT_SETS
    from model_sets import MODEL_SETS

    root      = Path(args.dir).resolve()
    csv_path  = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir    = root / "evaluation_results"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.model_set not in MODEL_SETS:
        print(f"Unknown model set '{args.model_set}'. Available: {list(MODEL_SETS.keys())}")
        return

    model_set   = MODEL_SETS[args.model_set]
    model_label = model_set["label"]
    set_keys    = list(PROMPT_SETS.keys())
    set_labels  = {k: PROMPT_SETS[k]["label"] for k in set_keys}

    # ── Load data ────────────────────────────────────────────────────────────
    data_rows = load_csv_rows(csv_path, args.rows)
    if not data_rows:
        print(f"No rows found in {csv_path}")
        return

    good_count = sum(1 for r in data_rows if r["human_label"] == "good")
    bad_count  = len(data_rows) - good_count

    n_jobs = len(data_rows) * len(set_keys)
    n_workers = min(args.workers, n_jobs)

    print(f"\n{'='*70}")
    print(f"  SYNTHETIC DATA EVALUATION")
    print(f"  CSV       : {csv_path.name}")
    print(f"  Rows      : {len(data_rows)}  (good={good_count}, bad={bad_count})")
    print(f"  Model Set : {model_label}")
    print(f"  Prompt Sets: {len(set_keys)}  →  {n_jobs} total jobs")
    print(f"  Workers   : {n_workers}")
    print(f"{'='*70}")
    for k in set_keys:
        print(f"  {k}  —  {PROMPT_SETS[k]['description']}")
    print()

    # ── Build jobs ───────────────────────────────────────────────────────────
    jobs: list[dict] = []
    for row in data_rows:
        for set_key in set_keys:
            jobs.append({
                "row": row,
                "set_key": set_key,
                "prompt_set": PROMPT_SETS[set_key],
                "model_set": model_set,
                "job_idx": len(jobs) + 1,
                "n_total": n_jobs,
            })

    wall_start = time.time()

    # ── Run in parallel ──────────────────────────────────────────────────────
    completed: list[dict] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one_job, job): job for job in jobs}
        for future in as_completed(futures):
            completed.append(future.result())

    wall_elapsed = round(time.time() - wall_start, 1)

    # ── Reassemble results into rows ─────────────────────────────────────────
    results_by_row: dict[int, dict] = {}
    for c in completed:
        idx = c["row_idx"]
        if idx not in results_by_row:
            results_by_row[idx] = {}
        results_by_row[idx][c["set_key"]] = {
            "result": c["result"], "error": c["error"], "elapsed": c["elapsed"],
        }

    for row in data_rows:
        row["results"] = results_by_row.get(row["row_idx"], {})

    # ── Metrics ──────────────────────────────────────────────────────────────
    overall_metrics: dict[str, dict[str, dict]] = {}
    for strat in STRATEGIES:
        overall_metrics[strat] = {
            k: compute_overall_metrics(data_rows, k, strat) for k in set_keys
        }

    section_metrics = {k: compute_section_accuracy(data_rows, k) for k in set_keys}
    runtime_stats   = compute_avg_runtime(data_rows, set_keys)

    # ── Print summary ────────────────────────────────────────────────────────
    sum_of_totals = sum(runtime_stats[k]["total"] for k in set_keys)
    print(f"\n{'='*70}")
    print(f"  WALL CLOCK : {wall_elapsed:.1f}s   (sequential would be ~{sum_of_totals:.0f}s)")
    if wall_elapsed > 0:
        print(f"  SPEEDUP    : {sum_of_totals / wall_elapsed:.1f}×")
    print(f"{'='*70}")

    print(f"\n{'='*70}")
    print("  AVERAGE RUNTIME PER JOB")
    print(f"{'='*70}")
    print(f"  {'Prompt Set':<32}  {'Avg':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*32}  {'-'*8}  {'-'*8}  {'-'*8}")
    for k in set_keys:
        rs = runtime_stats[k]
        print(f"  {k:<32}  {rs['mean']:7.1f}s  {rs['min']:7.1f}s  {rs['max']:7.1f}s")

    for strat in STRATEGIES:
        strat_label = "WEIGHTED ≥50%" if strat == "weighted" else "STRICT (all-pass)"
        print(f"\n{'='*70}")
        print(f"  {strat_label}")
        print(f"{'='*70}")
        print(f"  {'Prompt Set':<32}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  TP TN FP FN")
        print(f"  {'-'*32}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  ---------")
        for k in set_keys:
            m = overall_metrics[strat][k]
            print(f"  {k:<32}  {m['accuracy']*100:5.1f}%  {m['precision']*100:5.1f}%"
                  f"  {m['recall']*100:5.1f}%  {m['f1']*100:5.1f}%"
                  f"  {m['tp']:2} {m['tn']:2} {m['fp']:2} {m['fn']:2}")

    print(f"\n{'='*70}")
    print("  PER-SECTION ACCURACY")
    print(f"{'='*70}")
    for k in set_keys:
        print(f"\n  {set_labels[k]}")
        print(f"  {'Section':<10}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")
        print(f"  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
        for sec in ("plo", "methods", "results", "plan"):
            sm = section_metrics[k][sec]
            print(f"  {sec.upper():<10}  {sm['accuracy']*100:6.1f}%  {sm['precision']*100:6.1f}%"
                  f"  {sm['recall']*100:6.1f}%  {sm['f1']*100:6.1f}%")

    # ── Write files ──────────────────────────────────────────────────────────
    txt_path  = outdir / f"synth_eval_{timestamp}.txt"
    csv_path_out = outdir / f"synth_eval_{timestamp}.csv"
    json_path = outdir / f"synth_eval_{timestamp}.json"

    txt_path.write_text(
        build_txt(data_rows, set_keys, set_labels, model_label,
                  overall_metrics, section_metrics, runtime_stats),
        encoding="utf-8",
    )
    csv_path_out.write_text(
        build_csv(data_rows, set_keys),
        encoding="utf-8",
    )
    json_path.write_text(
        build_json(data_rows, set_keys, runtime_stats, model_label),
        encoding="utf-8",
    )

    print(f"\n  Results saved to:")
    print(f"    {txt_path}")
    print(f"    {csv_path_out}")
    print(f"    {json_path}  ← structured verdicts with reasons\n")


if __name__ == "__main__":
    main()
