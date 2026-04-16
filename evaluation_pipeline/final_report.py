"""
final_report.py — Combined 9-way ablation study report.

Reads three evaluation CSV files (one per model set, each covering 3 prompt sets)
and produces:

  evaluation_results/final_report_YYYYMMDD_HHMMSS.txt
  evaluation_results/final_report_YYYYMMDD_HHMMSS.csv

The 9 configurations (3 model sets × 3 prompt sets):
  MS1 (LLaMA-4-Maverick + GPT-5.4-mini) × [Prompt A, B, C]
  MS2 (o4-mini-evaluator + gpt-4.1-verifier) × [Prompt A, B, C]
  MS3 (DeepSeek-V3.2 + o4-mini-verifier)   × [Prompt A, B, C]

Usage:
  python final_report.py --ms1 path/to/ms1_eval.csv
                         --ms2 path/to/ms2_eval.csv
                         --ms3 path/to/ms3_eval.csv
  
  If paths are omitted the script auto-selects the latest eval_*.csv
  from each folder (using filename metadata written during each run).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import sys
from pathlib import Path

DEFAULT_THRESHOLD = 0.5
EVAL_DIR = Path(__file__).parent / "evaluation_results"

MODEL_SET_LABELS = {
    "MS1": "MS1 — LLaMA-4-Maverick + GPT-5.4-mini  (baseline)",
    "MS2": "MS2 — o4-mini-evaluator + gpt-4.1-verifier  (reasoning evaluator)",
    "MS3": "MS3 — DeepSeek-V3.2 + o4-mini-verifier  (reasoning verifier)",
}
PROMPT_LABELS = {
    "A": "Prompt A — Disqualifier-First",
    "B": "Prompt B — Chain-of-Thought Evidence Anchoring",
    "C": "Prompt C — Few-Shot Exemplar Comparison",
}


# ── CSV parsing ───────────────────────────────────────────────────────────────

def load_csv(path: Path, threshold: float) -> dict[str, dict]:
    """
    Read an evaluation CSV and return per-prompt metrics.
    Returns {prompt_key: {tp, tn, fp, fn, accuracy, precision, recall, f1,
                          section_pass, section_total, per_file}}
    """
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    results: dict[str, dict] = {}
    for prompt_key in ("A", "B", "C"):
        pred_col  = f"{prompt_key}_pred"
        match_col = f"{prompt_key}_match"
        plo_col   = f"{prompt_key}_plo"
        met_col   = f"{prompt_key}_methods"
        res_col   = f"{prompt_key}_results"
        pln_col   = f"{prompt_key}_plan"

        if pred_col not in (rows[0] if rows else {}):
            # Try lowercase or different naming
            pred_col  = next((c for c in rows[0] if c.lower().startswith(prompt_key.lower()) and "pred" in c), None)
            if not pred_col:
                continue

        tp = tn = fp = fn = 0
        section_pass = {"plo": 0, "methods": 0, "results": 0, "plan": 0}
        section_total = 0
        per_file = []

        for row in rows:
            label = row.get("human_label", "unknown")
            pred  = row.get(pred_col, "")
            if label == "unknown" or not pred:
                continue
            if   label == "good" and pred == "good": tp += 1
            elif label == "bad"  and pred == "bad":  tn += 1
            elif label == "bad"  and pred == "good": fp += 1
            elif label == "good" and pred == "bad":  fn += 1

            try:
                section_pass["plo"]     += int(row.get(plo_col, 0) or 0)
                section_pass["methods"] += int(row.get(met_col, 0) or 0)
                section_pass["results"] += int(row.get(res_col, 0) or 0)
                section_pass["plan"]    += int(row.get(pln_col, 0) or 0)
                section_total += 1
            except (ValueError, TypeError):
                pass

            per_file.append({
                "file":  row.get("file", ""),
                "label": label,
                "pred":  pred,
                "match": row.get(match_col, ""),
                "score": row.get(f"{prompt_key}_weighted_pct", "") or row.get(f"{prompt_key}_score_pct", ""),
            })

        total = tp + tn + fp + fn
        acc   = (tp + tn) / total    if total         else 0
        prec  = tp / (tp + fp)       if (tp + fp)     else 0
        rec   = tp / (tp + fn)       if (tp + fn)     else 0
        f1    = 2*prec*rec/(prec+rec) if (prec + rec) else 0

        results[prompt_key] = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "section_pass": section_pass, "section_total": section_total,
            "per_file": per_file,
        }
    return results


# ── Confusion matrix ──────────────────────────────────────────────────────────

def cm_str(m: dict) -> str:
    tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]
    return (f"                PREDICTED\n"
            f"                good   bad\n"
            f"  ACTUAL  good  [{tp:3d} | {fn:3d}]  TP|FN\n"
            f"           bad  [{fp:3d} | {tn:3d}]  FP|TN")


# ── Report builders ───────────────────────────────────────────────────────────

def build_txt(all_metrics: dict[str, dict[str, dict]]) -> str:
    """
    all_metrics = {
        "MS1": {"A": {tp,tn,...}, "B": {...}, "C": {...}},
        "MS2": {...},
        "MS3": {...},
    }
    """
    buf = io.StringIO()
    w   = buf.write
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    SEP = "=" * 100

    w(SEP + "\n")
    w("  ROAR PIPELINE — FINAL ABLATION STUDY REPORT\n")
    w("  9 Configurations: 3 Model Sets × 3 Prompt Sets\n")
    w(f"  Generated : {now}\n")
    w(f"  Threshold : 50% -> GOOD  |  below -> BAD\n")
    w(SEP + "\n\n")

    # ── Model set descriptions ────────────────────────────────────────────────
    w("MODEL SETS\n" + "-" * 70 + "\n")
    for ms, label in MODEL_SET_LABELS.items():
        w(f"  {label}\n")
    w("\nPROMPT SETS\n" + "-" * 70 + "\n")
    for pk, label in PROMPT_LABELS.items():
        w(f"  {label}\n")
    w("\n")

    # ── 9-cell metrics table ──────────────────────────────────────────────────
    w("METRICS — ALL 9 CONFIGURATIONS\n")
    w("-" * 100 + "\n")
    w(f"  {'Configuration':<42}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  TP  TN  FP  FN\n")
    w("-" * 100 + "\n")

    best_f1    = 0.0
    best_label = ""
    all_cells  = []

    for ms_key in ("MS1", "MS2", "MS3"):
        for pk in ("A", "B", "C"):
            m     = all_metrics[ms_key][pk]
            label = f"{ms_key} × Prompt {pk}"
            f1    = m["f1"]
            if f1 > best_f1:
                best_f1    = f1
                best_label = label
            marker = " <-- BEST" if f1 == best_f1 else ""
            w(f"  {label:<42}  {m['accuracy']*100:5.1f}%  {m['precision']*100:5.1f}%"
              f"  {m['recall']*100:5.1f}%  {f1*100:5.1f}%"
              f"  {m['tp']:3}  {m['tn']:3}  {m['fp']:3}  {m['fn']:3}{marker}\n")
            all_cells.append((label, m))
        w("-" * 100 + "\n")

    # Mark actual best after full scan
    best_f1 = max(c[1]["f1"] for c in all_cells)
    w("\n")

    # ── Confusion matrices ────────────────────────────────────────────────────
    w("CONFUSION MATRICES\n")
    for ms_key in ("MS1", "MS2", "MS3"):
        w(f"\n  {MODEL_SET_LABELS[ms_key]}\n")
        for pk in ("A", "B", "C"):
            m = all_metrics[ms_key][pk]
            w(f"  Prompt {pk}:\n")
            for line in cm_str(m).splitlines():
                w(f"    {line}\n")
        w("\n")

    # ── Per-section pass rates ────────────────────────────────────────────────
    w("PER-SECTION PASS RATES\n")
    w("-" * 100 + "\n")
    w(f"  {'Config':<30}  {'PLO':>8}  {'Methods':>8}  {'Results':>8}  {'Plan':>8}\n")
    w("-" * 100 + "\n")
    for ms_key in ("MS1", "MS2", "MS3"):
        for pk in ("A", "B", "C"):
            m   = all_metrics[ms_key][pk]
            tot = m["section_total"]
            sp  = m["section_pass"]
            pct = lambda s: f"{sp[s]}/{tot}={sp[s]/tot*100:.0f}%" if tot else "N/A"
            w(f"  {ms_key} × Prompt {pk:<20}  {pct('plo'):>8}  {pct('methods'):>8}  {pct('results'):>8}  {pct('plan'):>8}\n")
    w("\n")

    # ── Best configuration analysis ───────────────────────────────────────────
    w(SEP + "\n")
    w("BEST CONFIGURATION ANALYSIS\n")
    w(SEP + "\n\n")

    best_cells = [(lbl, m) for lbl, m in all_cells if m["f1"] == best_f1]
    for lbl, m in best_cells:
        w(f"  Winner : {lbl}\n")
        w(f"  F1-Score   : {m['f1']*100:.2f}%\n")
        w(f"  Accuracy   : {m['accuracy']*100:.2f}%\n")
        w(f"  Precision  : {m['precision']*100:.2f}%\n")
        w(f"  Recall     : {m['recall']*100:.2f}%\n")
        w(f"  TP/TN/FP/FN: {m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}\n\n")

    # ── Ranking table ─────────────────────────────────────────────────────────
    w("RANKING (by F1-Score)\n")
    w("-" * 60 + "\n")
    sorted_cells = sorted(all_cells, key=lambda x: x[1]["f1"], reverse=True)
    for rank, (lbl, m) in enumerate(sorted_cells, 1):
        w(f"  #{rank:2}  {lbl:<30}  F1={m['f1']*100:.1f}%  Acc={m['accuracy']*100:.1f}%"
          f"  Prec={m['precision']*100:.1f}%  Rec={m['recall']*100:.1f}%\n")
    w("\n")

    # ── Per-file comparison table (all 9 configs) ─────────────────────────────
    w(SEP + "\n")
    w("PER-FILE RESULTS — ALL 9 CONFIGURATIONS\n")
    w(SEP + "\n")
    w(f"  {'FILE':<46} {'LBL':>5}")
    for ms_key in ("MS1","MS2","MS3"):
        for pk in ("A","B","C"):
            hdr = f"{ms_key}{pk}"
            w(f"  {hdr:>7}")
    w("\n" + "-" * 100 + "\n")

    # Collect per-file data (use MS1-A as the file list reference)
    ref_files = all_metrics["MS1"]["A"]["per_file"]
    for ref_row in ref_files:
        fname = ref_row["file"][:45]
        label = ref_row["label"]
        line  = f"  {fname:<46} {label:>5}"
        for ms_key in ("MS1","MS2","MS3"):
            for pk in ("A","B","C"):
                pf  = all_metrics[ms_key][pk]["per_file"]
                hit = next((r for r in pf if r["file"] == ref_row["file"]), None)
                if hit:
                    ok = "OK" if hit["match"] in ("OK", "  OK") else "MISS"
                    line += f"  {ok:>7}"
                else:
                    line += f"  {'N/A':>7}"
        w(line + "\n")
    w("\n")

    w(SEP + "\n")
    w("END OF REPORT\n")
    return buf.getvalue()


def build_csv(all_metrics: dict[str, dict[str, dict]]) -> str:
    buf = io.StringIO()
    fieldnames = ["config", "model_set", "prompt_set",
                  "accuracy_pct", "precision_pct", "recall_pct", "f1_pct",
                  "tp", "tn", "fp", "fn", "total",
                  "plo_pass_pct", "methods_pass_pct", "results_pass_pct", "plan_pass_pct"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for ms_key in ("MS1","MS2","MS3"):
        for pk in ("A","B","C"):
            m   = all_metrics[ms_key][pk]
            tot = m["section_total"]
            sp  = m["section_pass"]
            writer.writerow({
                "config":           f"{ms_key}_Prompt{pk}",
                "model_set":        MODEL_SET_LABELS[ms_key],
                "prompt_set":       PROMPT_LABELS[pk],
                "accuracy_pct":     round(m["accuracy"]  * 100, 1),
                "precision_pct":    round(m["precision"] * 100, 1),
                "recall_pct":       round(m["recall"]    * 100, 1),
                "f1_pct":           round(m["f1"]        * 100, 1),
                "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
                "total": m["total"],
                "plo_pass_pct":     round(sp["plo"]     / tot * 100, 1) if tot else 0,
                "methods_pass_pct": round(sp["methods"] / tot * 100, 1) if tot else 0,
                "results_pass_pct": round(sp["results"] / tot * 100, 1) if tot else 0,
                "plan_pass_pct":    round(sp["plan"]    / tot * 100, 1) if tot else 0,
            })
    return buf.getvalue()


# ── Main ──────────────────────────────────────────────────────────────────────

def find_latest_eval_csv(tag: str) -> Path | None:
    """Find the most recently modified eval_*.csv in evaluation_results/."""
    candidates = sorted(EVAL_DIR.glob("eval_*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Combine 3 model-set evaluations into a final report")
    parser.add_argument("--ms1", metavar="CSV", help="Path to MS1 evaluation CSV")
    parser.add_argument("--ms2", metavar="CSV", help="Path to MS2 evaluation CSV")
    parser.add_argument("--ms3", metavar="CSV", help="Path to MS3 evaluation CSV")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    # ── Locate CSVs ───────────────────────────────────────────────────────────
    csv_paths: dict[str, Path] = {}
    if args.ms1: csv_paths["MS1"] = Path(args.ms1)
    if args.ms2: csv_paths["MS2"] = Path(args.ms2)
    if args.ms3: csv_paths["MS3"] = Path(args.ms3)

    if len(csv_paths) < 3:
        # Auto-detect: pick the 3 most recent eval_*.csv files
        all_csvs = sorted(EVAL_DIR.glob("eval_*.csv"), key=lambda p: p.stat().st_mtime)
        if len(all_csvs) < 3:
            print(f"Need 3 eval CSV files in {EVAL_DIR}. Found {len(all_csvs)}.")
            print("Pass --ms1 --ms2 --ms3 explicitly, or run all 3 evaluations first.")
            sys.exit(1)
        # Use the 3 most recent
        auto = all_csvs[-3:]
        keys = [k for k in ("MS1","MS2","MS3") if k not in csv_paths]
        for k, p in zip(keys, auto):
            csv_paths[k] = p
            print(f"  Auto-detected {k}: {p.name}")

    print(f"\nLoading evaluation results...")
    for ms_key, path in csv_paths.items():
        print(f"  {ms_key}: {path}")

    all_metrics: dict[str, dict] = {}
    for ms_key, path in csv_paths.items():
        all_metrics[ms_key] = load_csv(path, args.threshold)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  9-CONFIGURATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Configuration':<30}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

    best_f1 = 0.0
    best_lbl = ""
    for ms_key in ("MS1","MS2","MS3"):
        for pk in ("A","B","C"):
            m   = all_metrics[ms_key][pk]
            lbl = f"{ms_key} × Prompt {pk}"
            print(f"  {lbl:<30}  {m['accuracy']*100:5.1f}%  {m['precision']*100:5.1f}%"
                  f"  {m['recall']*100:5.1f}%  {m['f1']*100:5.1f}%")
            if m["f1"] > best_f1:
                best_f1, best_lbl = m["f1"], lbl

    print(f"\n  Best F1: {best_lbl}  (F1={best_f1*100:.1f}%)\n")

    # ── Write outputs ─────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = EVAL_DIR / f"final_report_{timestamp}.txt"
    csv_path = EVAL_DIR / f"final_report_{timestamp}.csv"

    txt_path.write_text(build_txt(all_metrics), encoding="utf-8")
    csv_path.write_text(build_csv(all_metrics), encoding="utf-8")

    print(f"  Reports saved to:")
    print(f"    {txt_path}")
    print(f"    {csv_path}\n")


if __name__ == "__main__":
    main()
