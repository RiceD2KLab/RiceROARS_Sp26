"""
model_evaluation_test.py — Ablation study across 3 model set configurations.

Runs every ROAR .docx through all three model sets using the BEST prompt set
(Set A — Disqualifier-First, chosen from the prompt ablation study) and
produces timestamped results in evaluation_results/.

  evaluation_results/model_eval_YYYYMMDD_HHMMSS.txt
  evaluation_results/model_eval_YYYYMMDD_HHMMSS.csv

Prerequisites
─────────────
  Deploy the following models on Azure before running:
    o4-mini       → Azure OpenAI (cognitiveservices)   [Set 2 evaluator + Set 3 verifier]
    gpt-4.1       → Azure OpenAI (cognitiveservices)   [Set 2 verifier]
    DeepSeek-V3.2 → Azure AI Foundry MaaS              [Set 3 evaluator]

  Update the deployment names in model_sets.py, then run:
    python model_evaluation_test.py

Usage:
  python model_evaluation_test.py
  python model_evaluation_test.py --threshold 0.5
  python model_evaluation_test.py --prompt-set A_DisqualifierFirst
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import sys
import time
from pathlib import Path
from typing import Optional

# ── Known labels and skip dirs (same as evaluation_test.py) ──────────────────
KNOWN_LABELS: dict[str, str] = {}
_SKIP_DIRS = {
    "qwen-server", ".venv", "venv", "__pycache__", "node_modules",
    "site-packages", "dist-info", "reports", "evaluation_results",
}
DEFAULT_THRESHOLD   = 0.5
DEFAULT_PROMPT_SET  = "A_DisqualifierFirst"   # winner from prompt ablation


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


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_pipeline(model_set: dict, prompt_set: dict):
    """
    Build a ROARPipeline configured for a specific model set + prompt set.

    For the "azure_openai" backend, the model set dict does NOT contain the
    credentials (they come from config.py).  For the "openai" backend it also
    defers to config.py.  This means:
      - Set 1 (baseline)   → reads config values exactly as production
      - Set 2 (o4-mini)    → reads AZURE_ENDPOINT/API_KEY from config
      - Set 3 (DeepSeek)   → reads EVALUATOR_API_BASE/KEY from config for
                              evaluator; AZURE_ENDPOINT/KEY for verifier
    """
    import config
    from pipeline.roar_pipeline import ROARPipeline
    from utils.llm_factory import Backend

    ev_cfg = model_set["evaluator"]
    ve_cfg = model_set["verifier"]

    return ROARPipeline(
        evaluator_model=ev_cfg["model"],
        verifier_model=ve_cfg["model"],
        evaluator_backend=ev_cfg["backend"],
        verifier_backend=ve_cfg["backend"],
        prompt_set=prompt_set,
    )


def run_with_model_set(path: Path, model_set: dict, prompt_set: dict) -> tuple:
    from pipeline.extractor import SectionExtractor

    pipeline = build_pipeline(model_set, prompt_set)
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
    section_pass  = {"plo": 0, "methods": 0, "results": 0, "plan": 0}
    section_total = 0

    for r in rows:
        label  = r["human_label"]
        result = r["results"].get(set_key, {}).get("result")
        if label == "unknown" or result is None:
            continue
        pred = "good" if result.weighted_score >= threshold else "bad"
        if   label == "good" and pred == "good": tp += 1
        elif label == "bad"  and pred == "bad":  tn += 1
        elif label == "bad"  and pred == "good": fp += 1
        elif label == "good" and pred == "bad":  fn += 1
        for sec in ("plo", "methods", "results", "plan"):
            section_pass[sec] += getattr(result.final_scores, sec)
        section_total += 1

    total = tp + tn + fp + fn
    acc   = (tp + tn) / total   if total            else 0
    prec  = tp / (tp + fp)      if (tp + fp)        else 0
    rec   = tp / (tp + fn)      if (tp + fn)        else 0
    f1    = 2*prec*rec/(prec+rec) if (prec + rec)   else 0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "section_pass": section_pass, "section_total": section_total,
    }


def confusion_matrix_str(m: dict) -> str:
    tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]
    return "\n".join([
        "                  PREDICTED",
        "                  good    bad",
        f"  ACTUAL  good  [ {tp:3d}  | {fn:3d} ]  <- TP | FN",
        f"           bad  [ {fp:3d}  | {tn:3d} ]  <- FP | TN",
    ])


# ── Report builders ───────────────────────────────────────────────────────────

def trunc(t: str, n: int = 180) -> str:
    t = t.strip().replace("\n", " ")
    return t[:n] + ("..." if len(t) > n else "")


def build_txt(rows, set_keys, model_sets_cfg, threshold, metrics, prompt_label) -> str:
    buf = io.StringIO()
    w   = buf.write
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    w("=" * 100 + "\n")
    w("  ROAR PIPELINE — MODEL SET ABLATION STUDY\n")
    w(f"  Generated    : {now}\n")
    w(f"  Prompt set   : {prompt_label}  (winner of prompt ablation; held fixed)\n")
    w(f"  Threshold    : {threshold*100:.0f}% -> GOOD  |  below -> BAD\n")
    w(f"  Total files  : {len(rows)}\n")
    w("=" * 100 + "\n\n")

    w("MODEL SETS TESTED\n")
    w("-" * 70 + "\n")
    for k in set_keys:
        ms = model_sets_cfg[k]
        w(f"  {k}\n")
        w(f"    {ms['description']}\n")
        w(f"    Evaluator : {ms['evaluator']['model']}  (backend: {ms['evaluator']['backend']})\n")
        w(f"    Verifier  : {ms['verifier']['model']}   (backend: {ms['verifier']['backend']})\n\n")

    # Metrics table
    w("METRICS COMPARISON\n")
    w("-" * 100 + "\n")
    w(f"  {'Metric':<22}")
    for k in set_keys:
        w(f"  {k:<30}")
    w("\n" + "-" * 100 + "\n")
    for metric, label in [("accuracy","Accuracy"),("precision","Precision"),
                          ("recall","Recall (Sensitivity)"),("f1","F1-Score")]:
        w(f"  {label:<22}")
        for k in set_keys:
            w(f"  {metrics[k][metric]*100:5.1f}%{'':<24}")
        w("\n")
    w("-" * 100 + "\n")
    w(f"  {'TP/TN/FP/FN':<22}")
    for k in set_keys:
        m = metrics[k]
        w(f"  {m['tp']}TP {m['tn']}TN {m['fp']}FP {m['fn']}FN{'':<15}")
    w("\n\n")

    # Confusion matrices
    w("CONFUSION MATRICES\n")
    for k in set_keys:
        w(f"\n  {model_sets_cfg[k]['label']}\n")
        for line in confusion_matrix_str(metrics[k]).splitlines():
            w(f"  {line}\n")
    w("\n")

    # Per-section pass rates
    w("PER-SECTION PASS RATES\n")
    w("-" * 80 + "\n")
    w(f"  {'Section':<12}")
    for k in set_keys:
        w(f"  {k:<30}")
    w("\n" + "-" * 80 + "\n")
    for sec in ("plo", "methods", "results", "plan"):
        w(f"  {sec.upper():<12}")
        for k in set_keys:
            m   = metrics[k]
            tot = m["section_total"]
            pct = m["section_pass"][sec] / tot * 100 if tot else 0
            w(f"  {m['section_pass'][sec]}/{tot} = {pct:5.1f}%{'':<17}")
        w("\n")
    w("\n")

    # Per-file table
    w("PER-FILE RESULTS\n")
    sep = "-" * 102
    w(sep + "\n")
    hdr = f"  {'FILE':<44} {'LABEL':>7}"
    for k in set_keys:
        short = k.split("_")[0]
        hdr += f"  {short+' SCORE':>10}  {short+' PRED':>8}  {short+' OK?':>6}"
    w(hdr + "\n" + sep + "\n")
    for r in rows:
        label = r["human_label"]
        line  = f"  {r['file']:<44} {label:>7}"
        for k in set_keys:
            res    = r["results"].get(k, {})
            result = res.get("result")
            if not result:
                line += f"  {'ERR':>10}  {'ERR':>8}  {'':>6}"
            else:
                score = result.weighted_score
                pred  = "good" if score >= threshold else "bad"
                ok    = ("OK" if (label != "unknown" and pred == label)
                         else ("MISS" if label != "unknown" else "--"))
                line += f"  {score*100:8.0f}%  {pred:>8}  {ok:>6}"
        w(line + "\n")
    w(sep + "\n\n")

    # Recommendation
    best = max(set_keys, key=lambda k: metrics[k]["f1"])
    bm   = metrics[best]
    w("RECOMMENDATION\n")
    w("-" * 60 + "\n")
    w(f"  Best F1: {model_sets_cfg[best]['label']}\n")
    w(f"  F1={bm['f1']*100:.1f}%  Accuracy={bm['accuracy']*100:.1f}%  "
      f"Precision={bm['precision']*100:.1f}%  Recall={bm['recall']*100:.1f}%\n\n")

    # Detailed audit
    w("=" * 100 + "\n")
    w("DETAILED PER-FILE AUDIT\n")
    w("=" * 100 + "\n\n")
    for r in rows:
        w("-" * 100 + "\n")
        w(f"FILE  : {r['file']}\nLABEL : {r['human_label']}\n\n")
        for k in set_keys:
            res    = r["results"].get(k, {})
            result = res.get("result")
            err    = res.get("error")
            elapsed = res.get("elapsed", 0)
            ms     = model_sets_cfg[k]
            w(f"  [{ms['label']}]  ({elapsed}s)\n")
            if err or not result:
                w(f"    ERROR: {err}\n\n")
                continue
            score = result.weighted_score
            pred  = "good" if score >= threshold else "bad"
            ok    = ("OK" if (r["human_label"] != "unknown" and pred == r["human_label"])
                     else ("MISS" if r["human_label"] != "unknown" else "--"))
            s = result.final_scores
            w(f"    Scores  : PLO={s.plo}  Methods={s.methods}  "
              f"Results={s.results}  Plan={s.plan}\n")
            w(f"    Weighted: {score*100:.0f}%  Pred={pred.upper()}  Match={ok}\n")
            w(f"    Consist : {'Yes' if result.consistent else 'No'}  "
              f"Iter={result.iterations}\n")
            if result.reasoning:
                for sec in ("plo", "methods", "results", "plan"):
                    txt = getattr(result.reasoning, sec, "")
                    if txt:
                        w(f"    [{sec.upper():8}] {trunc(txt)}\n")
            w("\n")

    w("=" * 100 + "\nEND OF REPORT\n")
    return buf.getvalue()


def build_csv(rows, set_keys, threshold) -> str:
    buf = io.StringIO()
    base = ["file", "folder", "human_label"]
    cols = []
    for k in set_keys:
        s = k.split("_")[0]
        cols += [f"{s}_plo", f"{s}_methods", f"{s}_results", f"{s}_plan",
                 f"{s}_score_pct", f"{s}_pred", f"{s}_match",
                 f"{s}_consistent", f"{s}_iterations", f"{s}_error"]
    writer = csv.DictWriter(buf, fieldnames=base + cols, lineterminator="\n")
    writer.writeheader()
    for r in rows:
        row = {"file": r["file"], "folder": r["folder"], "human_label": r["human_label"]}
        for k in set_keys:
            s      = k.split("_")[0]
            res    = r["results"].get(k, {})
            result = res.get("result")
            err    = res.get("error", "")
            if result and not err:
                score = result.weighted_score
                pred  = "good" if score >= threshold else "bad"
                ok    = ("OK" if (r["human_label"] != "unknown" and pred == r["human_label"])
                         else ("MISS" if r["human_label"] != "unknown" else "--"))
                fs    = result.final_scores
                row.update({
                    f"{s}_plo": fs.plo, f"{s}_methods": fs.methods,
                    f"{s}_results": fs.results, f"{s}_plan": fs.plan,
                    f"{s}_score_pct": round(score * 100, 1),
                    f"{s}_pred": pred, f"{s}_match": ok,
                    f"{s}_consistent": result.consistent,
                    f"{s}_iterations": result.iterations, f"{s}_error": "",
                })
            else:
                for f in [f"{s}_plo", f"{s}_methods", f"{s}_results", f"{s}_plan",
                          f"{s}_score_pct", f"{s}_pred", f"{s}_match",
                          f"{s}_consistent", f"{s}_iterations"]:
                    row[f] = ""
                row[f"{s}_error"] = err or "no result"
        writer.writerow(row)
    return buf.getvalue()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",         default=".", metavar="PATH")
    parser.add_argument("--threshold",   type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--prompt-set",  default=DEFAULT_PROMPT_SET,
                        dest="prompt_set_key",
                        help=f"Prompt set to use (default: {DEFAULT_PROMPT_SET})")
    args = parser.parse_args()

    from model_sets  import MODEL_SETS
    from prompt_sets import PROMPT_SETS

    if args.prompt_set_key not in PROMPT_SETS:
        print(f"Unknown prompt set: {args.prompt_set_key}")
        print(f"Available: {list(PROMPT_SETS.keys())}")
        sys.exit(1)

    prompt_set   = PROMPT_SETS[args.prompt_set_key]
    prompt_label = f"{args.prompt_set_key} — {prompt_set['description']}"
    set_keys     = list(MODEL_SETS.keys())
    root         = Path(args.dir).resolve()
    timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir       = root / "evaluation_results"
    outdir.mkdir(parents=True, exist_ok=True)
    pairs        = find_files(root)

    if not pairs:
        print(f"No .docx files found under {root}")
        return

    n_total = len(pairs) * len(set_keys)
    print(f"\n{'='*70}")
    print(f"  ROAR MODEL SET ABLATION — {len(pairs)} files x {len(set_keys)} model sets = {n_total} runs")
    print(f"  Prompt set (fixed) : {args.prompt_set_key}")
    print(f"{'='*70}")
    for k in set_keys:
        ms = MODEL_SETS[k]
        print(f"  {k}: {ms['evaluator']['model']} + {ms['verifier']['model']}")
    print()

    rows: list[dict] = []
    for file_idx, (path, label) in enumerate(pairs, 1):
        print(f"\n[{file_idx:02}/{len(pairs):02}] {path.name}  (label: {label})")
        row = {"file": path.name, "folder": path.parent.name,
               "human_label": label, "results": {}}
        for set_key in set_keys:
            ms = MODEL_SETS[set_key]
            print(f"  -> {set_key} ...", end="", flush=True)
            result, error, elapsed = run_with_model_set(path, ms, prompt_set)
            if error:
                print(f"  ERROR ({elapsed}s)")
            else:
                score = result.weighted_score
                pred  = "good" if score >= args.threshold else "bad"
                ok    = ("OK" if (label != "unknown" and pred == label)
                         else ("MISS" if label != "unknown" else "--"))
                print(f"  {score*100:.0f}%  {pred}  [{ok}]  ({elapsed}s)")
            row["results"][set_key] = {"result": result, "error": error, "elapsed": elapsed}
        rows.append(row)

    all_metrics = {k: compute_metrics(rows, k, args.threshold) for k in set_keys}

    print(f"\n{'='*70}\n  RESULTS SUMMARY\n{'='*70}")
    print(f"  {'Model Set':<36}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  TP TN FP FN")
    print(f"  {'-'*36}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  ---------")
    for k in set_keys:
        m = all_metrics[k]
        print(f"  {k:<36}  {m['accuracy']*100:5.1f}%  {m['precision']*100:5.1f}%"
              f"  {m['recall']*100:5.1f}%  {m['f1']*100:5.1f}%"
              f"  {m['tp']:2} {m['tn']:2} {m['fp']:2} {m['fn']:2}")
    best = max(set_keys, key=lambda k: all_metrics[k]["f1"])
    print(f"\n  Best F1: {MODEL_SETS[best]['label']}  "
          f"(F1={all_metrics[best]['f1']*100:.1f}%)\n")

    txt_path = outdir / f"model_eval_{timestamp}.txt"
    csv_path = outdir / f"model_eval_{timestamp}.csv"
    txt_path.write_text(
        build_txt(rows, set_keys, MODEL_SETS, args.threshold, all_metrics, prompt_label),
        encoding="utf-8",
    )
    csv_path.write_text(build_csv(rows, set_keys, args.threshold), encoding="utf-8")
    print(f"  Results saved to:\n    {txt_path}\n    {csv_path}\n")


if __name__ == "__main__":
    main()
