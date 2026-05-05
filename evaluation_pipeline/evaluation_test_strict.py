"""
evaluation_test_strict.py — Compare two classification strategies:

  1. WEIGHTED (current)   :  weighted_score >= 50%  →  GOOD
  2. STRICT  (all-pass)   :  every section == 1     →  GOOD
                             any section  == 0      →  BAD  (report which ones failed)

Runs every ROAR .docx through all three prompt sets (A, B, C) and reports metrics
for BOTH strategies side-by-side so you can see the trade-off.

The prompts / pipeline / scoring are IDENTICAL to evaluation_test.py.
Only the final good/bad decision rule changes.

Outputs:
  evaluation_results/strict_eval_YYYYMMDD_HHMMSS.txt    Full report
  evaluation_results/strict_eval_YYYYMMDD_HHMMSS.csv    Per-file scores + reason column
  evaluation_results/strict_eval_YYYYMMDD_HHMMSS.json   Structured verdicts (prediction,
                                                         failed_sections, reason, section_reasons)
                                                         ready for downstream consumption

Usage:
  python evaluation_test_strict.py
  python evaluation_test_strict.py --dir d:/449pipeline
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

from models.schemas import PipelineResult, SectionScores

# ── Known labels ─────────────────────────────────────────────────────────────
KNOWN_LABELS: dict[str, str] = {}
WEIGHTED_THRESHOLD = 0.5


# ── File discovery (same as evaluation_test.py) ─────────────────────────────

_SKIP_DIRS = {
    "qwen-server", ".venv", "venv", "__pycache__", "node_modules",
    "site-packages", "dist-info", "reports", "evaluation_results",
}

def find_files(root: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*.docx")):
        parts_lower = {p.lower() for p in path.parts}
        if parts_lower & _SKIP_DIRS:
            continue
        parent = path.parent.name.lower()
        if "good" in parent:
            label = "good"
        elif "bad" in parent:
            label = "bad"
        else:
            label = KNOWN_LABELS.get(path.name, "unknown")
        pairs.append((path, label))
    return pairs


# ── Pipeline runner (same as evaluation_test.py) ────────────────────────────

def build_pipeline(model_set: dict | None, prompt_set: dict):
    from pipeline.roar_pipeline import ROARPipeline
    if model_set is None:
        return ROARPipeline(prompt_set=prompt_set)
    ev_cfg = model_set["evaluator"]
    ve_cfg = model_set["verifier"]
    return ROARPipeline(
        evaluator_model=ev_cfg["model"],
        verifier_model=ve_cfg["model"],
        evaluator_backend=ev_cfg["backend"],
        verifier_backend=ve_cfg["backend"],
        prompt_set=prompt_set,
    )


def run_with_prompt_set(path: Path, prompt_set: dict,
                        model_set: dict | None = None) -> tuple:
    """Return (PipelineResult | None, error_str | None, elapsed_sec)."""
    from pipeline.extractor import SectionExtractor

    pipeline = build_pipeline(model_set, prompt_set)
    start    = time.time()
    try:
        sections = SectionExtractor.extract_from_docx(path)
        result   = pipeline.run(pre_sections=sections)
        return result, None, round(time.time() - start, 1)
    except Exception as exc:
        return None, str(exc)[:200], round(time.time() - start, 1)


# ── Classification strategies ────────────────────────────────────────────────

def classify_weighted(result: PipelineResult, threshold: float = WEIGHTED_THRESHOLD) -> str:
    return "good" if result.weighted_score >= threshold else "bad"


def classify_strict(result: PipelineResult) -> str:
    """ALL four sections must be 1 (PASS) for the ROAR to be 'good'."""
    s = result.final_scores
    if s.plo == 1 and s.methods == 1 and s.results == 1 and s.plan == 1:
        return "good"
    return "bad"


def failed_sections(scores: SectionScores) -> list[str]:
    """Return the names of sections that scored 0."""
    return [sec for sec in ("plo", "methods", "results", "plan")
            if getattr(scores, sec) == 0]


def build_verdict(result: PipelineResult, human_label: str) -> dict:
    """Build a structured verdict dict for a single pipeline run.

    The returned dict is JSON-serialisable and contains a top-level `reason`
    string summarising why the ROAR failed (empty string when it passes).
    """
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
        "scores": {
            "plo": s.plo,
            "methods": s.methods,
            "results": s.results,
            "plan": s.plan,
        },
        "failed_sections": fails,
        "reason": reason,
        "section_reasons": section_reasons,
        "consistent": result.consistent,
        "iterations": result.iterations,
    }


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict], set_key: str,
                    classify_fn) -> dict:
    tp = tn = fp = fn_ = 0
    section_pass = {"plo": 0, "methods": 0, "results": 0, "plan": 0}
    section_total = 0

    for r in rows:
        label  = r["human_label"]
        result = r["results"].get(set_key, {}).get("result")
        if label == "unknown" or result is None:
            continue
        pred = classify_fn(result)
        if   label == "good" and pred == "good": tp  += 1
        elif label == "bad"  and pred == "bad":  tn  += 1
        elif label == "bad"  and pred == "good": fp  += 1
        elif label == "good" and pred == "bad":  fn_ += 1

        for sec in ("plo", "methods", "results", "plan"):
            section_pass[sec] += getattr(result.final_scores, sec)
        section_total += 1

    total = tp + tn + fp + fn_
    acc   = (tp + tn) / total if total else 0
    prec  = tp / (tp + fp) if (tp + fp) else 0
    rec   = tp / (tp + fn_) if (tp + fn_) else 0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn_, "total": total,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "section_pass": section_pass, "section_total": section_total,
    }


# ── ASCII confusion matrix ──────────────────────────────────────────────────

def confusion_matrix_str(m: dict) -> str:
    tp, tn, fp, fn_ = m["tp"], m["tn"], m["fp"], m["fn"]
    return "\n".join([
        "                  PREDICTED",
        "                  good    bad",
        f"  ACTUAL  good  [ {tp:3d}  | {fn_:3d} ]  ← TP | FN",
        f"           bad  [ {fp:3d}  | {tn:3d} ]  ← FP | TN",
    ])


# ── Report builders ─────────────────────────────────────────────────────────

def trunc(t: str, n: int = 200) -> str:
    t = t.strip().replace("\n", " ")
    return t[:n] + ("…" if len(t) > n else "")


STRATEGIES = ["weighted", "strict"]


def compute_avg_runtime(rows: list[dict], set_keys: list[str]) -> dict[str, dict]:
    """Return {set_key: {"mean": float, "min": float, "max": float, "total": float, "count": int}}."""
    stats: dict[str, dict] = {}
    for k in set_keys:
        times = []
        for r in rows:
            res = r["results"].get(k, {})
            if res.get("result") is not None and res.get("error") is None:
                times.append(res["elapsed"])
        if times:
            stats[k] = {
                "mean": round(sum(times) / len(times), 1),
                "min": min(times),
                "max": max(times),
                "total": round(sum(times), 1),
                "count": len(times),
            }
        else:
            stats[k] = {"mean": 0, "min": 0, "max": 0, "total": 0, "count": 0}
    return stats


def build_txt(rows: list[dict], set_keys: list[str],
              set_labels: dict[str, str],
              metrics: dict[str, dict[str, dict]],
              runtime_stats: dict[str, dict]) -> str:
    buf = io.StringIO()
    w   = buf.write
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    w("=" * 110 + "\n")
    w("  ROAR CLASSIFICATION STRATEGY COMPARISON\n")
    w("  Weighted (≥50%) vs. Strict (all sections must pass)\n")
    w(f"  Generated: {now}\n")
    import config as _cfg
    _ev = f"{_cfg.EVALUATOR_MODEL}  (backend: {_cfg.EVALUATOR_BACKEND})"
    _ve_deploy = getattr(_cfg, "VERIFIER_AZURE_DEPLOYMENT", _cfg.VERIFIER_MODEL)
    _ve = f"{_ve_deploy}  (backend: {_cfg.VERIFIER_BACKEND})"
    w(f"  Evaluator: {_ev}\n")
    w(f"  Verifier : {_ve}\n")
    w(f"  Files    : {len(rows)}\n")
    w("=" * 110 + "\n\n")

    # ── Strategy explanation ─────────────────────────────────────────────────
    w("STRATEGY DEFINITIONS\n")
    w("-" * 70 + "\n")
    w("  WEIGHTED : pred = GOOD if  (PLO×25% + Methods×30% + Results×30% + Plan×15%) ≥ 50%\n")
    w("             A single failing section can still yield GOOD if the rest pass.\n")
    w("  STRICT   : pred = GOOD only if ALL four sections score 1 (PASS).\n")
    w("             Any single 0 → BAD.  Reports WHICH section(s) failed.\n")
    w("             Trade-off: higher precision (fewer bad ROARs slip through),\n")
    w("                        possibly lower recall (more good ROARs flagged for review).\n\n")

    # ── Prompt sets ──────────────────────────────────────────────────────────
    w("PROMPT SETS TESTED\n")
    w("-" * 60 + "\n")
    for k in set_keys:
        from prompt_sets import PROMPT_SETS
        ps = PROMPT_SETS[k]
        w(f"  {k}\n    {ps['description']}\n")
    w("\n")

    # ── Average runtime ──────────────────────────────────────────────────────
    w("AVERAGE RUNTIME PER PROMPT SET\n")
    w("─" * 90 + "\n")
    w(f"  {'Prompt Set':<32}  {'Avg':>8}  {'Min':>8}  {'Max':>8}  {'Total':>9}  {'Runs':>5}\n")
    w("─" * 90 + "\n")
    for k in set_keys:
        rs = runtime_stats[k]
        w(f"  {k:<32}  {rs['mean']:7.1f}s  {rs['min']:7.1f}s  {rs['max']:7.1f}s  {rs['total']:8.1f}s  {rs['count']:5d}\n")
    w("\n")

    # ── Metrics comparison ───────────────────────────────────────────────────
    for strat in STRATEGIES:
        strat_label = "WEIGHTED (≥50%)" if strat == "weighted" else "STRICT (all-must-pass)"
        w(f"{'─' * 110}\n")
        w(f"METRICS — {strat_label}\n")
        w(f"{'─' * 110}\n")
        w(f"  {'Metric':<22}")
        for k in set_keys:
            w(f"  {k:<30}")
        w("\n" + "─" * 110 + "\n")

        for metric, label in [
            ("accuracy",  "Accuracy"),
            ("precision", "Precision"),
            ("recall",    "Recall (Sensitivity)"),
            ("f1",        "F1-Score"),
        ]:
            w(f"  {label:<22}")
            for k in set_keys:
                v = metrics[strat][k][metric]
                w(f"  {v*100:5.1f}%{'':<24}")
            w("\n")

        w("─" * 110 + "\n")
        w(f"  {'TP / TN / FP / FN':<22}")
        for k in set_keys:
            m = metrics[strat][k]
            w(f"  {m['tp']}TP {m['tn']}TN {m['fp']}FP {m['fn']}FN{'':<17}")
        w("\n\n")

    # ── Side-by-side delta ───────────────────────────────────────────────────
    w("=" * 110 + "\n")
    w("STRATEGY DELTA  (strict minus weighted)\n")
    w("=" * 110 + "\n")
    w(f"  {'Prompt Set':<30}  {'ΔAcc':>7}  {'ΔPrec':>7}  {'ΔRec':>7}  {'ΔF1':>7}  {'ΔFP':>5}  {'ΔFN':>5}\n")
    w("-" * 90 + "\n")
    for k in set_keys:
        mw = metrics["weighted"][k]
        ms = metrics["strict"][k]
        da = (ms["accuracy"]  - mw["accuracy"])  * 100
        dp = (ms["precision"] - mw["precision"]) * 100
        dr = (ms["recall"]    - mw["recall"])    * 100
        df = (ms["f1"]        - mw["f1"])        * 100
        dfp = ms["fp"] - mw["fp"]
        dfn = ms["fn"] - mw["fn"]
        w(f"  {k:<30}  {da:+6.1f}%  {dp:+6.1f}%  {dr:+6.1f}%  {df:+6.1f}%  {dfp:+4d}   {dfn:+4d}\n")
    w("\n")
    w("  Negative ΔFP = fewer bad ROARs slipping through (good).\n")
    w("  Positive ΔFN = more good ROARs flagged for review (trade-off).\n\n")

    # ── Confusion matrices ───────────────────────────────────────────────────
    for strat in STRATEGIES:
        strat_label = "WEIGHTED" if strat == "weighted" else "STRICT"
        w(f"CONFUSION MATRICES — {strat_label}\n")
        for k in set_keys:
            w(f"\n  {set_labels[k]}\n")
            for line in confusion_matrix_str(metrics[strat][k]).splitlines():
                w(f"  {line}\n")
        w("\n")

    # ── Per-file comparison table ────────────────────────────────────────────
    w("=" * 110 + "\n")
    w("PER-FILE RESULTS  (weighted pred → strict pred)\n")
    w("=" * 110 + "\n")

    for k in set_keys:
        short = k.split("_")[0]
        w(f"\n  [{set_labels[k]}]\n")
        w(f"  {'FILE':<44} {'LABEL':>6}  {'SCORE':>6}  {'W-PRED':>7}  {'S-PRED':>7}  {'FAILED SECTIONS':<30}  {'MATCH':>5}\n")
        w("  " + "─" * 108 + "\n")

        for r in rows:
            label  = r["human_label"]
            res    = r["results"].get(k, {})
            result = res.get("result")
            err    = res.get("error")

            if err or result is None:
                w(f"  {r['file']:<44} {label:>6}  {'ERR':>6}  {'ERR':>7}  {'ERR':>7}  {'':<30}  {'':<5}\n")
                continue

            w_pred = classify_weighted(result)
            s_pred = classify_strict(result)
            fails  = failed_sections(result.final_scores)
            fails_str = ", ".join(s.upper() for s in fails) if fails else "—"
            ok = "OK" if (label != "unknown" and s_pred == label) else (
                 "MISS" if label != "unknown" else "--")

            w(f"  {r['file']:<44} {label:>6}  {result.weighted_score*100:5.0f}%  {w_pred:>7}  {s_pred:>7}  {fails_str:<30}  {ok:>5}\n")
        w("\n")

    # ── Per-file detailed audit (strict focus) ───────────────────────────────
    w("=" * 110 + "\n")
    w("DETAILED AUDIT — STRICT STRATEGY (why each ROAR was marked BAD)\n")
    w("=" * 110 + "\n\n")

    for r in rows:
        w("─" * 110 + "\n")
        w(f"FILE  : {r['file']}\n")
        w(f"LABEL : {r['human_label']}\n\n")

        for k in set_keys:
            res    = r["results"].get(k, {})
            result = res.get("result")
            err    = res.get("error")
            elapsed = res.get("elapsed", 0)

            w(f"  [{set_labels[k]}]  ({elapsed}s)\n")

            if err or result is None:
                w(f"    ERROR: {err}\n\n")
                continue

            s      = result.final_scores
            s_pred = classify_strict(result)
            w_pred = classify_weighted(result)
            fails  = failed_sections(s)

            w(f"    Scores  : PLO={s.plo}  Methods={s.methods}  Results={s.results}  Plan={s.plan}\n")
            w(f"    Weighted: {result.weighted_score*100:.0f}%  (w-pred={w_pred.upper()})   Strict: s-pred={s_pred.upper()}\n")

            if fails:
                w(f"    FAILED  : {', '.join(sec.upper() for sec in fails)}\n")
                if result.reasoning:
                    for sec in fails:
                        txt = getattr(result.reasoning, sec, "")
                        if txt:
                            w(f"      → {sec.upper()}: {trunc(txt, 150)}\n")
            else:
                w(f"    All sections PASS.\n")

            w(f"    Consist : {'Yes' if result.consistent else 'No'}  Iter={result.iterations}\n\n")

    w("=" * 110 + "\n")
    w("END OF REPORT\n")
    return buf.getvalue()


def build_csv(rows: list[dict], set_keys: list[str]) -> str:
    buf = io.StringIO()
    base_fields = ["file", "folder", "human_label"]
    score_fields = []
    for k in set_keys:
        short = k.split("_")[0]
        score_fields += [
            f"{short}_plo", f"{short}_methods", f"{short}_results", f"{short}_plan",
            f"{short}_weighted_pct",
            f"{short}_weighted_pred", f"{short}_weighted_match",
            f"{short}_strict_pred", f"{short}_strict_match",
            f"{short}_failed_sections", f"{short}_reason",
            f"{short}_consistent", f"{short}_iterations",
            f"{short}_elapsed_sec", f"{short}_error",
        ]
    writer = csv.DictWriter(buf, fieldnames=base_fields + score_fields, lineterminator="\n")
    writer.writeheader()

    for r in rows:
        row = {"file": r["file"], "folder": r["folder"], "human_label": r["human_label"]}
        for k in set_keys:
            short  = k.split("_")[0]
            res    = r["results"].get(k, {})
            result = res.get("result")
            err    = res.get("error", "")
            elapsed = res.get("elapsed", 0)
            if result and not err:
                verdict = build_verdict(result, r["human_label"])
                s = result.final_scores
                w_pred = classify_weighted(result)
                label  = r["human_label"]
                w_ok = ("OK" if (label != "unknown" and w_pred == label)
                        else ("MISS" if label != "unknown" else "--"))
                s_ok = ("OK" if (label != "unknown" and verdict["prediction"] == label)
                        else ("MISS" if label != "unknown" else "--"))
                row.update({
                    f"{short}_plo": s.plo, f"{short}_methods": s.methods,
                    f"{short}_results": s.results, f"{short}_plan": s.plan,
                    f"{short}_weighted_pct": round(result.weighted_score * 100, 1),
                    f"{short}_weighted_pred": w_pred, f"{short}_weighted_match": w_ok,
                    f"{short}_strict_pred": verdict["prediction"],
                    f"{short}_strict_match": s_ok,
                    f"{short}_failed_sections": "|".join(verdict["failed_sections"]),
                    f"{short}_reason": verdict["reason"],
                    f"{short}_consistent": result.consistent,
                    f"{short}_iterations": result.iterations,
                    f"{short}_elapsed_sec": elapsed,
                    f"{short}_error": "",
                })
            else:
                for f in [f"{short}_plo", f"{short}_methods", f"{short}_results",
                          f"{short}_plan", f"{short}_weighted_pct",
                          f"{short}_weighted_pred", f"{short}_weighted_match",
                          f"{short}_strict_pred", f"{short}_strict_match",
                          f"{short}_failed_sections", f"{short}_reason",
                          f"{short}_consistent", f"{short}_iterations",
                          f"{short}_elapsed_sec"]:
                    row[f] = ""
                row[f"{short}_error"] = err or "no result"
        writer.writerow(row)

    return buf.getvalue()


def build_json(rows: list[dict], set_keys: list[str],
               runtime_stats: dict[str, dict]) -> str:
    """Build a JSON array of per-file verdicts across all prompt sets.

    Each entry contains a `verdicts` dict keyed by prompt set, with each
    verdict having `prediction`, `passed`, `failed_sections`, `reason`,
    and full `section_reasons` — ready for downstream consumption.
    """
    output = []
    for r in rows:
        entry: dict = {
            "file": r["file"],
            "folder": r["folder"],
            "human_label": r["human_label"],
            "verdicts": {},
        }
        for k in set_keys:
            res    = r["results"].get(k, {})
            result = res.get("result")
            err    = res.get("error")
            elapsed = res.get("elapsed", 0)

            if err or result is None:
                entry["verdicts"][k] = {
                    "error": err or "no result",
                    "elapsed_sec": elapsed,
                }
            else:
                v = build_verdict(result, r["human_label"])
                v["elapsed_sec"] = elapsed
                entry["verdicts"][k] = v

        output.append(entry)

    wrapper = {
        "generated": datetime.datetime.now().isoformat(),
        "strategy": "strict_all_must_pass",
        "runtime_stats": runtime_stats,
        "files": output,
    }
    return json.dumps(wrapper, indent=2, ensure_ascii=False)


# ── Parallel batch runner ────────────────────────────────────────────────────

_print_lock = threading.Lock()

def _safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def _run_one_job(job: dict) -> dict:
    """Execute a single (file, prompt_set) pair and return the result dict.

    Designed to be called from a thread pool. Each invocation creates its own
    pipeline instance, so there is no shared mutable state.
    """
    path     = job["path"]
    set_key  = job["set_key"]
    ps       = job["prompt_set"]
    ms       = job.get("model_set")
    label    = job["label"]
    job_idx  = job["job_idx"]
    n_total  = job["n_total"]

    _safe_print(f"  [{job_idx:3d}/{n_total}]  {path.name}  ×  {set_key} ...", flush=True)
    result, error, elapsed = run_with_prompt_set(path, ps, model_set=ms)

    if error:
        _safe_print(f"  [{job_idx:3d}/{n_total}]  {path.name}  ×  {set_key}  ERROR ({elapsed}s)")
    else:
        s_pred = classify_strict(result)
        ok = "OK" if (label != "unknown" and s_pred == label) else (
             "MISS" if label != "unknown" else "--")
        _safe_print(
            f"  [{job_idx:3d}/{n_total}]  {path.name}  ×  {set_key}"
            f"  {result.weighted_score*100:.0f}%  s={s_pred}  [{ok}]  ({elapsed}s)"
        )

    return {
        "file": path.name, "folder": path.parent.name,
        "label": label, "set_key": set_key,
        "result": result, "error": error, "elapsed": elapsed,
    }


DEFAULT_WORKERS = 6

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Compare weighted (≥50%) vs strict (all-must-pass) classification strategies.")
    parser.add_argument("--dir",     default=".", metavar="PATH")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, metavar="N",
                        help=f"Number of parallel threads (default {DEFAULT_WORKERS}). "
                             "Each thread runs one (file × prompt_set) job.")
    parser.add_argument("--model-set", default=None,
                        help="Model set key from model_sets.py (e.g. MS2_o4mini_GPT41). "
                             "If omitted, uses config.py defaults.")
    args = parser.parse_args()

    from prompt_sets import PROMPT_SETS

    model_set = None
    model_label = None
    if args.model_set:
        from model_sets import MODEL_SETS
        if args.model_set not in MODEL_SETS:
            print(f"Unknown model set '{args.model_set}'. Available: {list(MODEL_SETS.keys())}")
            return
        model_set = MODEL_SETS[args.model_set]
        model_label = model_set["label"]

    root      = Path(args.dir).resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir    = root / "evaluation_results"
    outdir.mkdir(parents=True, exist_ok=True)

    pairs    = find_files(root)
    set_keys = list(PROMPT_SETS.keys())
    set_labels = {k: PROMPT_SETS[k]["label"] for k in set_keys}

    if not pairs:
        print(f"No .docx files found under {root}")
        return

    # Build flat list of jobs: every (file, prompt_set) combination
    jobs: list[dict] = []
    for path, label in pairs:
        for set_key in set_keys:
            jobs.append({
                "path": path, "label": label,
                "set_key": set_key, "prompt_set": PROMPT_SETS[set_key],
                "model_set": model_set,
                "job_idx": len(jobs) + 1, "n_total": len(pairs) * len(set_keys),
            })

    n_workers = min(args.workers, len(jobs))
    n_total   = len(jobs)

    print(f"\n{'='*70}")
    print(f"  STRICT vs WEIGHTED — {len(pairs)} files × {len(set_keys)} prompt sets = {n_total} runs")
    if model_label:
        print(f"  Model Set: {model_label}")
    print(f"  Workers: {n_workers} parallel threads")
    print(f"{'='*70}")
    for k in set_keys:
        print(f"  {k}  —  {PROMPT_SETS[k]['description']}")
    print()

    wall_start = time.time()

    # ── Run all jobs in parallel ─────────────────────────────────────────────
    completed_jobs: list[dict] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one_job, job): job for job in jobs}
        for future in as_completed(futures):
            completed_jobs.append(future.result())

    wall_elapsed = round(time.time() - wall_start, 1)

    # ── Reassemble into per-file rows (same structure as before) ─────────────
    row_map: dict[str, dict] = {}
    for j in completed_jobs:
        key = (j["file"], j["folder"])
        if key not in row_map:
            row_map[key] = {
                "file": j["file"], "folder": j["folder"],
                "human_label": j["label"], "results": {},
            }
        row_map[key]["results"][j["set_key"]] = {
            "result": j["result"], "error": j["error"], "elapsed": j["elapsed"],
        }

    rows = sorted(row_map.values(), key=lambda r: r["file"])

    # ── Compute metrics for both strategies ──────────────────────────────────
    all_metrics: dict[str, dict[str, dict]] = {}
    all_metrics["weighted"] = {
        k: compute_metrics(rows, k, classify_weighted) for k in set_keys
    }
    all_metrics["strict"] = {
        k: compute_metrics(rows, k, classify_strict) for k in set_keys
    }

    runtime_stats = compute_avg_runtime(rows, set_keys)

    # ── Print summary ────────────────────────────────────────────────────────
    sum_of_totals = sum(runtime_stats[k]["total"] for k in set_keys)
    print(f"\n{'='*70}")
    print(f"  WALL CLOCK : {wall_elapsed:.1f}s   (sequential would be ~{sum_of_totals:.0f}s)")
    print(f"  SPEEDUP    : {sum_of_totals / wall_elapsed:.1f}×" if wall_elapsed > 0 else "")
    print(f"  WORKERS    : {n_workers}")
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
            m = all_metrics[strat][k]
            print(f"  {k:<32}  {m['accuracy']*100:5.1f}%  {m['precision']*100:5.1f}%"
                  f"  {m['recall']*100:5.1f}%  {m['f1']*100:5.1f}%"
                  f"  {m['tp']:2} {m['tn']:2} {m['fp']:2} {m['fn']:2}")

    print(f"\n{'='*70}")
    print("  DELTA  (strict − weighted)")
    print(f"{'='*70}")
    print(f"  {'Prompt Set':<32}  {'ΔAcc':>7}  {'ΔPrec':>7}  {'ΔRec':>7}  {'ΔF1':>7}")
    print(f"  {'-'*32}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for k in set_keys:
        mw = all_metrics["weighted"][k]
        ms = all_metrics["strict"][k]
        print(f"  {k:<32}  {(ms['accuracy']-mw['accuracy'])*100:+6.1f}%"
              f"  {(ms['precision']-mw['precision'])*100:+6.1f}%"
              f"  {(ms['recall']-mw['recall'])*100:+6.1f}%"
              f"  {(ms['f1']-mw['f1'])*100:+6.1f}%")

    # ── Write files ──────────────────────────────────────────────────────────
    txt_path  = outdir / f"strict_eval_{timestamp}.txt"
    csv_path  = outdir / f"strict_eval_{timestamp}.csv"
    json_path = outdir / f"strict_eval_{timestamp}.json"

    txt_path.write_text(
        build_txt(rows, set_keys, set_labels, all_metrics, runtime_stats),
        encoding="utf-8",
    )
    csv_path.write_text(
        build_csv(rows, set_keys),
        encoding="utf-8",
    )
    json_path.write_text(
        build_json(rows, set_keys, runtime_stats),
        encoding="utf-8",
    )

    print(f"\n  Results saved to:")
    print(f"    {txt_path}")
    print(f"    {csv_path}")
    print(f"    {json_path}  ← structured verdicts with reasons\n")


if __name__ == "__main__":
    main()
