#!/usr/bin/env python3
"""
Post-process GPT-4o-mini results v2:
- Extract core answer WITHOUT using GT (blind extraction)
- Remove empty/timeout predictions from denominator
- Recompute metrics
"""
import json, re, numpy as np
from collections import Counter
from pathlib import Path

def normalize_answer(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())

def compute_em(p, g): return float(normalize_answer(p) == normalize_answer(g))
def compute_f1(p, g):
    pt, gt = normalize_answer(p).split(), normalize_answer(g).split()
    c = Counter(pt) & Counter(gt); n = sum(c.values())
    if n == 0: return 0.0
    pr = n/len(pt) if pt else 0; rc = n/len(gt) if gt else 0
    return 2*pr*rc/(pr+rc) if pr+rc > 0 else 0.0

def extract_blind(pred):
    """Extract core answer from verbose response WITHOUT knowing GT."""
    pred = pred.strip()
    if not pred:
        return ""

    # Remove leading "A: " or "Answer: "
    pred = re.sub(r'^(?:A:|Answer:)\s*', '', pred, flags=re.IGNORECASE).strip()

    # Yes/No: if starts with Yes/No, return that
    m = re.match(r'^(yes|no)\b', pred, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # "The answer is X" pattern
    m = re.search(r'(?:the answer is|answer is)\s+["\']?(.+?)["\']?[\.,!]?\s*$', pred, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip('.')

    # "It's a/an X" → X
    m = re.match(r"^It'?s\s+(?:a|an)\s+(.+?)[\.,!]", pred, re.IGNORECASE)
    if m and len(m.group(1).split()) <= 4:
        return m.group(1).strip()

    # "A/An X." → X (short answers)
    m = re.match(r'^(?:A|An)\s+(.+?)[\.,!]?\s*$', pred, re.IGNORECASE)
    if m and len(m.group(1).split()) <= 3:
        return m.group(1).strip()

    # Quoted content — often the actual answer
    quoted = re.findall(r'"([^"]+)"', pred)
    if quoted and len(quoted[0].split()) <= 6:
        return quoted[0]

    # Take text before first comma or period (first clause)
    first_clause = re.split(r'[,.]', pred)[0].strip()

    # Remove common prefixes from first clause
    first_clause = re.sub(r'^(?:The|This|That|These|Those)\s+\w+\s+(?:is|are|was|were)\s+', '', first_clause, flags=re.IGNORECASE).strip()

    if len(first_clause.split()) <= 5:
        return first_clause

    # If still long, take last meaningful phrase (often the answer comes after "is")
    m = re.search(r'\b(?:is|are|was|were)\s+(.+?)[\.,!]', pred, re.IGNORECASE)
    if m and len(m.group(1).split()) <= 5:
        return m.group(1).strip().strip('"\'')

    # Final fallback: first 5 words
    return ' '.join(pred.split()[:5])


def extract_vqa_answer(pred):
    """Extract short VQA answer (typically 1-2 words)."""
    pred = pred.strip()
    if not pred:
        return ""

    # "It's a X" → X
    m = re.match(r"^It'?s\s+(?:a|an\s+)?(.+?)[\.,!]?\s*$", pred, re.IGNORECASE)
    if m and len(m.group(1).split()) <= 3:
        return m.group(1).strip()

    # "A/An X." → X
    m = re.match(r'^(?:A|An)\s+(.+?)[\.,!]?\s*$', pred, re.IGNORECASE)
    if m and len(m.group(1).split()) <= 3:
        return m.group(1).strip()

    # Short answer already
    if len(pred.split()) <= 3:
        return pred.rstrip('.')

    # Take first noun phrase (up to first verb or comma)
    first = re.split(r'[,.]', pred)[0].strip()
    # Remove sentence starters
    first = re.sub(r'^(?:The|This|That|It|There)\s+(?:is|are|was|were|appears?|seems?|looks?)\s+(?:a|an|the)?\s*', '', first, flags=re.IGNORECASE).strip()
    if len(first.split()) <= 3:
        return first

    # Look for key patterns
    m = re.search(r'\b(?:is|are)\s+(?:a|an)?\s*(.+?)[\.,!]', pred, re.IGNORECASE)
    if m and len(m.group(1).split()) <= 3:
        return m.group(1).strip()

    # Fallback: first 2 words
    return ' '.join(pred.split()[:2])


def postprocess_benchmark(name, ckpt_path, metric_type, extract_fn=extract_blind):
    print(f"\n=== {name} Post-processing ===")
    ckpt = json.load(open(ckpt_path))

    results_b, results_m = [], []
    skipped = 0

    for k, v in ckpt.items():
        pred_b_raw = v.get('pred_b', '')
        pred_m_raw = v.get('pred_m', '')

        # Skip if both empty
        if not pred_b_raw.strip() and not pred_m_raw.strip():
            skipped += 1
            continue

        if metric_type == 'em_f1':
            gt = str(v['gt'])
            if pred_b_raw.strip():
                core_b = extract_fn(pred_b_raw)
                results_b.append({'em': compute_em(core_b, gt), 'f1': compute_f1(core_b, gt),
                                  'raw': pred_b_raw, 'core': core_b, 'gt': gt})
            if pred_m_raw.strip():
                core_m = extract_fn(pred_m_raw)
                results_m.append({'em': compute_em(core_m, gt), 'f1': compute_f1(core_m, gt),
                                  'raw': pred_m_raw, 'core': core_m, 'gt': gt})

        elif metric_type == 'accuracy':
            # ScienceQA - already letter-based
            if pred_b_raw.strip():
                results_b.append({'correct': v['correct_b']})
            if pred_m_raw.strip():
                results_m.append({'correct': v['correct_m']})

        elif metric_type == 'vqa_acc':
            gt_answers = v['gt'] if isinstance(v['gt'], list) else [v['gt']]
            if pred_b_raw.strip():
                core_b = extract_fn(pred_b_raw)
                correct = float(any(normalize_answer(core_b) == normalize_answer(a) for a in gt_answers))
                results_b.append({'correct': correct, 'raw': pred_b_raw, 'core': core_b})
            if pred_m_raw.strip():
                core_m = extract_fn(pred_m_raw)
                correct = float(any(normalize_answer(core_m) == normalize_answer(a) for a in gt_answers))
                results_m.append({'correct': correct, 'raw': pred_m_raw, 'core': core_m})

    n_total = len(ckpt)
    print(f"  Total: {n_total}, Skipped (both empty): {skipped}")
    print(f"  Valid baseline: {len(results_b)}, Valid MC: {len(results_m)}")

    summary = {"n_total": n_total, "n_valid_b": len(results_b), "n_valid_m": len(results_m)}

    if metric_type == 'em_f1':
        em_b = np.mean([v['em'] for v in results_b]) * 100
        f1_b = np.mean([v['f1'] for v in results_b]) * 100
        em_m = np.mean([v['em'] for v in results_m]) * 100
        f1_m = np.mean([v['f1'] for v in results_m]) * 100
        print(f"  Baseline: EM={em_b:.2f}% F1={f1_b:.2f}%")
        print(f"  MemCanvas: EM={em_m:.2f}% F1={f1_m:.2f}%")
        summary['baseline'] = {'em': em_b, 'f1': f1_b}
        summary['memcanvas'] = {'em': em_m, 'f1': f1_m}

        # Show some examples
        print(f"\n  Examples (first 5):")
        for r in results_b[:5]:
            print(f"    GT='{r['gt'][:40]}' raw='{r['raw'][:50]}' → core='{r['core'][:40]}' EM={r['em']}")

    elif metric_type == 'accuracy':
        acc_b = np.mean([v['correct'] for v in results_b]) * 100
        acc_m = np.mean([v['correct'] for v in results_m]) * 100
        print(f"  Baseline: {acc_b:.2f}%")
        print(f"  MemCanvas: {acc_m:.2f}%")
        summary['baseline'] = {'accuracy': acc_b}
        summary['memcanvas'] = {'accuracy': acc_m}

    elif metric_type == 'vqa_acc':
        acc_b = np.mean([v['correct'] for v in results_b]) * 100
        acc_m = np.mean([v['correct'] for v in results_m]) * 100
        print(f"  Baseline: {acc_b:.2f}%")
        print(f"  MemCanvas: {acc_m:.2f}%")
        summary['baseline'] = {'accuracy': acc_b}
        summary['memcanvas'] = {'accuracy': acc_m}

        print(f"\n  Examples (first 5):")
        for r in results_b[:5]:
            print(f"    raw='{r['raw'][:50]}' → core='{r['core'][:30]}' correct={r['correct']}")

    return summary


if __name__ == "__main__":
    all_results = {}

    all_results['hotpotqa'] = postprocess_benchmark(
        'HotpotQA', '/home/cyf/codex/gpt4omini_eval/hotpotqa/checkpoint.json',
        'em_f1', extract_blind)

    all_results['scienceqa'] = postprocess_benchmark(
        'ScienceQA', '/home/cyf/codex/gpt4omini_eval/scienceqa/checkpoint.json',
        'accuracy', extract_blind)

    all_results['okvqa'] = postprocess_benchmark(
        'OK-VQA', '/home/cyf/codex/gpt4omini_eval/okvqa/checkpoint.json',
        'vqa_acc', extract_vqa_answer)

    all_results['mmqa'] = postprocess_benchmark(
        'MMQA', '/home/cyf/codex/gpt4omini_eval/mmqa/checkpoint.json',
        'em_f1', extract_blind)

    # ScienceQA per-subject
    ckpt = json.load(open('/home/cyf/codex/gpt4omini_eval/scienceqa/checkpoint.json'))
    print("\n=== ScienceQA Per-Subject ===")
    for subj in ['natural science', 'social science', 'language science']:
        vals_b = [v['correct_b'] for v in ckpt.values() if v.get('subject','') == subj and v.get('pred_b','').strip()]
        vals_m = [v['correct_m'] for v in ckpt.values() if v.get('subject','') == subj and v.get('pred_m','').strip()]
        if vals_b:
            print(f"  {subj}: b={np.mean(vals_b)*100:.2f}% m={np.mean(vals_m)*100:.2f}%")
            all_results['scienceqa'][f'{subj}_b'] = np.mean(vals_b)*100
            all_results['scienceqa'][f'{subj}_m'] = np.mean(vals_m)*100

    out = Path('/home/cyf/codex/gpt4omini_eval/postprocessed_v2.json')
    json.dump(all_results, open(out, 'w'), indent=2)
    print(f"\nSaved to {out}")
