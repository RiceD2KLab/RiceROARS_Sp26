"""
evaluation_test.py — Ablation study across 3 prompt engineering strategies.

Runs every ROAR .docx through all three prompt sets (A, B, C) and produces:

  evaluation_results/eval_YYYYMMDD_HHMMSS.txt   Full report with per-file
                                                 step-by-step details + metrics
  evaluation_results/eval_YYYYMMDD_HHMMSS.csv   Per-file scores for all sets

Metrics computed per prompt set:
  Confusion matrix (TP / TN / FP / FN)
  Accuracy, Precision, Recall, F1-score
  Per-section pass rates

Usage:
  python evaluation_test.py
  python evaluation_test.py --threshold 0.5
  python evaluation_test.py --dir d:/449pipeline
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import time
from pathlib import Path
from typing import Optional

# ── Known labels for files outside labelled folders ──────────────────────────
# All ROAR files are now in good_roars/ or bad_roars/ — no overrides needed.
KNOWN_LABELS: dict[str, str] = {}
DEFAULT_THRESHOLD = 0.5


# ── File discovery ────────────────────────────────────────────────────────────

# Folder names that are NOT ROAR document folders — skip files inside these
_SKIP_DIRS = {
    "qwen-server", ".venv", "venv", "__pycache__", "node_modules",
    "site-packages", "dist-info", "reports", "evaluation_results",
}

def find_files(root: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*.docx")):
        # Skip any file whose path passes through a non-ROAR directory
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


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_with_prompt_set(path: Path, prompt_set: dict) -> tuple:
    """Return (PipelineResult | None, error_str | None, elapsed_sec)."""
    from pipeline.extractor import SectionExtractor
    from pipeline.roar_pipeline import ROARPipeline

    pipeline = ROARPipeline(prompt_set=prompt_set)
    start    = time.time()
    try:
        sections = SectionExtractor.extract_from_docx(path)
        result   = pipeline.run(pre_sections=sections)
        return result, None, round(time.time() - start, 1)
    except Exception as exc:
        return None, str(exc)[:200], round(time.time() - start, 1)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict], set_key: str, threshold: float) -> dict:
    tp = tn = fp = fn = 0
    section_pass = {"plo": 0, "methods": 0, "results": 0, "plan": 0}
    section_total = 0

    for r in rows:
        label  = r["human_label"]
        result = r["results"].get(set_key, {}).get("result")
        if label == "unknown" or result is None:
            continue
        pred = "good" if result.weighted_score >= threshold else "bad"
        if label == "good" and pred == "good": tp += 1
        elif label == "bad"  and pred == "bad":  tn += 1
        elif label == "bad"  and pred == "good": fp += 1
        elif label == "good" and pred == "bad":  fn += 1

        for sec in ("plo", "methods", "results", "plan"):
            section_pass[sec] += getattr(result.final_scores, sec)
        section_total += 1

    total   = tp + tn + fp + fn
    acc     = (tp + tn) / total if total else 0
    prec    = tp / (tp + fp) if (tp + fp) else 0
    rec     = tp / (tp + fn) if (tp + fn) else 0
    f1      = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "section_pass": section_pass,
        "section_total": section_total,
    }


# ── ASCII confusion matrix ────────────────────────────────────────────────────

def confusion_matrix_str(m: dict) -> str:
    tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]
    lines = [
        "                  PREDICTED",
        "                  good    bad",
        f"  ACTUAL  good  [ {tp:3d}  | {fn:3d} ]  ← TP | FN",
        f"           bad  [ {fp:3d}  | {tn:3d} ]  ← FP | TN",
    ]
    return "\n".join(lines)


# ── Report builders ───────────────────────────────────────────────────────────

def trunc(t: str, n: int = 200) -> str:
    t = t.strip().replace("\n", " ")
    return t[:n] + ("…" if len(t) > n else "")


def build_txt(rows: list[dict], set_keys: list[str],
              set_labels: dict[str, str], threshold: float,
              metrics: dict[str, dict]) -> str:
    buf = io.StringIO()
    w   = buf.write
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    w("=" * 100 + "\n")
    w("  ROAR PIPELINE ABLATION STUDY — PROMPT ENGINEERING COMPARISON\n")
    w(f"  Generated   : {now}\n")
    import config as _cfg
    _ev = f"{_cfg.EVALUATOR_MODEL}  (backend: {_cfg.EVALUATOR_BACKEND})"
    _ve_deploy = getattr(_cfg, 'VERIFIER_AZURE_DEPLOYMENT', _cfg.VERIFIER_MODEL)
    _ve = f"{_ve_deploy}  (backend: {_cfg.VERIFIER_BACKEND})"
    w(f"  Evaluator   : {_ev}\n")
    w(f"  Verifier    : {_ve}\n")
    w(f"  Threshold   : {threshold*100:.0f}% → GOOD  |  below → BAD\n")
    w(f"  Total files : {len(rows)}\n")
    w("=" * 100 + "\n\n")

    # ── Prompt set descriptions ───────────────────────────────────────────────
    w("PROMPT SETS TESTED\n")
    w("-" * 60 + "\n")
    for k in set_keys:
        from prompt_sets import PROMPT_SETS
        ps = PROMPT_SETS[k]
        w(f"  {k}\n")
        w(f"    {ps['description']}\n")
    w("\n")

    # ── Metrics comparison table ──────────────────────────────────────────────
    w("METRICS COMPARISON\n")
    w("─" * 100 + "\n")
    w(f"  {'Metric':<20}")
    for k in set_keys:
        w(f"  {k:<30}")
    w("\n")
    w("─" * 100 + "\n")

    for metric, label in [
        ("accuracy",  "Accuracy"),
        ("precision", "Precision"),
        ("recall",    "Recall (Sensitivity)"),
        ("f1",        "F1-Score"),
    ]:
        w(f"  {label:<20}")
        for k in set_keys:
            v = metrics[k][metric]
            w(f"  {v*100:5.1f}%{'':<24}")
        w("\n")

    w("─" * 100 + "\n")
    w(f"  {'TP / TN / FP / FN':<20}")
    for k in set_keys:
        m = metrics[k]
        w(f"  {m['tp']}TP {m['tn']}TN {m['fp']}FP {m['fn']}FN{'':<17}")
    w("\n\n")

    # ── Confusion matrices ────────────────────────────────────────────────────
    w("CONFUSION MATRICES\n")
    for k in set_keys:
        w(f"\n  {set_labels[k]}\n")
        for line in confusion_matrix_str(metrics[k]).splitlines():
            w(f"  {line}\n")

    w("\n")

    # ── Per-section pass rates ────────────────────────────────────────────────
    w("PER-SECTION PASS RATES\n")
    w("─" * 80 + "\n")
    w(f"  {'Section':<12}")
    for k in set_keys:
        w(f"  {k:<30}")
    w("\n")
    w("─" * 80 + "\n")
    for sec in ("plo", "methods", "results", "plan"):
        w(f"  {sec.upper():<12}")
        for k in set_keys:
            m   = metrics[k]
            tot = m["section_total"]
            pct = m["section_pass"][sec] / tot * 100 if tot else 0
            w(f"  {m['section_pass'][sec]}/{tot} = {pct:5.1f}%{'':<18}")
        w("\n")
    w("\n")

    # ── Summary comparison table ──────────────────────────────────────────────
    w("PER-FILE RESULTS COMPARISON\n")
    sep = "─" * 100
    w(sep + "\n")
    hdr = f"  {'FILE':<44} {'LABEL':>7}"
    for k in set_keys:
        short = k.split("_")[0]
        hdr += f"  {short+' SCORE':>10}  {short+' PRED':>8}  {short+' OK?':>6}"
    w(hdr + "\n")
    w(sep + "\n")

    for r in rows:
        label = r["human_label"]
        line  = f"  {r['file']:<44} {label:>7}"
        for k in set_keys:
            res = r["results"].get(k, {})
            result = res.get("result")
            err    = res.get("error")
            if err or result is None:
                line += f"  {'ERR':>10}  {'ERR':>8}  {'':>6}"
            else:
                score = result.weighted_score
                pred  = "good" if score >= threshold else "bad"
                ok    = "OK" if (label != "unknown" and pred == label) else ("MISS" if label != "unknown" else "--")
                line += f"  {score*100:8.0f}%  {pred:>8}  {ok:>6}"
        w(line + "\n")
    w(sep + "\n\n")

    # ── Recommendation ────────────────────────────────────────────────────────
    best_key = max(set_keys, key=lambda k: metrics[k]["f1"])
    best_m   = metrics[best_key]
    w("RECOMMENDATION\n")
    w("-" * 60 + "\n")
    w(f"  Best F1-Score: {set_labels[best_key]}  ({best_m['f1']*100:.1f}%)\n")
    w(f"  Accuracy     : {best_m['accuracy']*100:.1f}%\n")
    w(f"  Precision    : {best_m['precision']*100:.1f}%\n")
    w(f"  Recall       : {best_m['recall']*100:.1f}%\n\n")

    # ── Per-file detailed audit ───────────────────────────────────────────────
    w("=" * 100 + "\n")
    w("DETAILED PER-FILE AUDIT (all prompt sets)\n")
    w("=" * 100 + "\n\n")

    for r in rows:
        w("─" * 100 + "\n")
        w(f"FILE    : {r['file']}\n")
        w(f"LABEL   : {r['human_label']}\n\n")

        for k in set_keys:
            res    = r["results"].get(k, {})
            result = res.get("result")
            err    = res.get("error")
            elapsed = res.get("elapsed", 0)

            w(f"  [{set_labels[k]}]  ({elapsed}s)\n")

            if err or result is None:
                w(f"    ERROR: {err}\n\n")
                continue

            score = result.weighted_score
            pred  = "good" if score >= threshold else "bad"
            ok    = ("OK" if (r["human_label"] != "unknown" and pred == r["human_label"])
                     else ("MISS" if r["human_label"] != "unknown" else "--"))

            s = result.final_scores
            w(f"    Scores  : PLO={s.plo}  Methods={s.methods}  Results={s.results}  Plan={s.plan}\n")
            w(f"    Weighted: {score*100:.0f}%  Pred={pred.upper()}  Match={ok}\n")
            w(f"    Consist : {'Yes' if result.consistent else 'No'}  Iter={result.iterations}\n")
            if result.reasoning:
                for sec in ("plo", "methods", "results", "plan"):
                    txt = getattr(result.reasoning, sec, "")
                    if txt:
                        w(f"    [{sec.upper():8}] {trunc(txt, 150)}\n")
            w("\n")

    w("=" * 100 + "\n")
    w("END OF REPORT\n")
    return buf.getvalue()


def build_csv(rows: list[dict], set_keys: list[str], threshold: float) -> str:
    buf = io.StringIO()
    base_fields = ["file", "folder", "human_label"]
    score_fields = []
    for k in set_keys:
        short = k.split("_")[0]
        score_fields += [
            f"{short}_plo", f"{short}_methods", f"{short}_results", f"{short}_plan",
            f"{short}_weighted_pct", f"{short}_pred", f"{short}_match",
            f"{short}_consistent", f"{short}_iterations", f"{short}_error",
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
            if result and not err:
                score = result.weighted_score
                pred  = "good" if score >= threshold else "bad"
                ok    = ("OK" if (r["human_label"] != "unknown" and pred == r["human_label"])
                         else ("MISS" if r["human_label"] != "unknown" else "--"))
                s = result.final_scores
                row.update({
                    f"{short}_plo": s.plo, f"{short}_methods": s.methods,
                    f"{short}_results": s.results, f"{short}_plan": s.plan,
                    f"{short}_weighted_pct": round(score * 100, 1),
                    f"{short}_pred": pred, f"{short}_match": ok,
                    f"{short}_consistent": result.consistent,
                    f"{short}_iterations": result.iterations,
                    f"{short}_error": "",
                })
            else:
                for f in [f"{short}_plo", f"{short}_methods", f"{short}_results",
                          f"{short}_plan", f"{short}_weighted_pct", f"{short}_pred",
                          f"{short}_match", f"{short}_consistent", f"{short}_iterations"]:
                    row[f] = ""
                row[f"{short}_error"] = err or "no result"
        writer.writerow(row)

    return buf.getvalue()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",       default=".", metavar="PATH")
    parser.add_argument("--threshold", type=float,  default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    from prompt_sets import PROMPT_SETS

    root      = Path(args.dir).resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir    = root / "evaluation_results"
    outdir.mkdir(parents=True, exist_ok=True)

    pairs    = find_files(root)
    set_keys = list(PROMPT_SETS.keys())  # ["A_...", "B_...", "C_..."]
    set_labels = {k: PROMPT_SETS[k]["label"] for k in set_keys}

    if not pairs:
        print(f"No .docx files found under {root}")
        return

    n_total = len(pairs) * len(set_keys)
    print(f"\n{'='*70}")
    print(f"  ROAR ABLATION STUDY — {len(pairs)} files × {len(set_keys)} prompt sets = {n_total} runs")
    print(f"{'='*70}")
    for k in set_keys:
        print(f"  {k}  —  {PROMPT_SETS[k]['description']}")
    print()

    rows: list[dict] = []

    for file_idx, (path, label) in enumerate(pairs, 1):
        print(f"\n[{file_idx:02}/{len(pairs):02}] {path.name}  (label: {label})")
        row = {"file": path.name, "folder": path.parent.name,
               "human_label": label, "results": {}}

        for set_key in set_keys:
            ps = PROMPT_SETS[set_key]
            print(f"  → {set_key} ...", end="", flush=True)
            result, error, elapsed = run_with_prompt_set(path, ps)

            if error:
                print(f"  ERROR ({elapsed}s)")
            else:
                score = result.weighted_score
                pred  = "good" if score >= args.threshold else "bad"
                ok    = ("OK" if (label != "unknown" and pred == label)
                         else ("MISS" if label != "unknown" else "--"))
                print(f"  {score*100:.0f}%  {pred}  [{ok}]  ({elapsed}s)")

            row["results"][set_key] = {
                "result": result, "error": error, "elapsed": elapsed
            }

        rows.append(row)

    # ── Compute metrics ───────────────────────────────────────────────────────
    all_metrics = {k: compute_metrics(rows, k, args.threshold) for k in set_keys}

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Prompt Set':<32}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  TP TN FP FN")
    print(f"  {'-'*32}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  ---------")
    for k in set_keys:
        m = all_metrics[k]
        print(f"  {k:<32}  {m['accuracy']*100:5.1f}%  {m['precision']*100:5.1f}%"
              f"  {m['recall']*100:5.1f}%  {m['f1']*100:5.1f}%"
              f"  {m['tp']:2} {m['tn']:2} {m['fp']:2} {m['fn']:2}")

    best = max(set_keys, key=lambda k: all_metrics[k]["f1"])
    print(f"\n  Best F1: {set_labels[best]}  (F1={all_metrics[best]['f1']*100:.1f}%)\n")

    # ── Write files ───────────────────────────────────────────────────────────
    txt_path = outdir / f"eval_{timestamp}.txt"
    csv_path = outdir / f"eval_{timestamp}.csv"

    txt_path.write_text(
        build_txt(rows, set_keys, set_labels, args.threshold, all_metrics),
        encoding="utf-8",
    )
    csv_path.write_text(
        build_csv(rows, set_keys, args.threshold),
        encoding="utf-8",
    )

    print(f"  Results saved to:")
    print(f"    {txt_path}")
    print(f"    {csv_path}\n")


if __name__ == "__main__":
    main()
