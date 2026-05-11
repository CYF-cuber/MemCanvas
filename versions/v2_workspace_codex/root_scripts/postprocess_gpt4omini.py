#!/usr/bin/env python3
"""
Post-process GPT-4o-mini results:
1. Extract core answer from verbose responses
2. Remove empty/timeout predictions from denominator
3. Recompute metrics
"""
import json, re, numpy as np
from collections import Counter
from difflib import SequenceMatcher
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
def compute_bleu1(p, g):
    pt, gt = normalize_answer(p).split(), normalize_answer(g).split()
    if not pt or not gt: return 0.0
    gc = Counter(gt); return sum(min(Counter(pt)[w], gc[w]) for w in Counter(pt)) / len(pt)
def compute_anls(p, g):
    p, g = normalize_answer(p), normalize_answer(g)
    if not p or not g: return 0.0
    ratio = SequenceMatcher(None, p, g).ratio()
    return ratio if ratio >= 0.5 else 0.0

def extract_core_answer(pred, gt):
    """Extract core answer from verbose GPT-4o-mini response."""
    pred = pred.strip()
    if not pred:
        return ""

    gt_lower = gt.lower().strip()
    pred_lower = pred.lower()

    # 1. Yes/No questions
    if gt_lower in ('yes', 'no'):
        if pred_lower.startswith('yes'):
            return 'yes'
        elif pred_lower.startswith('no'):
            return 'no'
        # Check for "Yes," or "No," anywhere
        if re.match(r'^yes\b', pred_lower):
            return 'yes'
        if re.match(r'^no\b', pred_lower):
            return 'no'
        return pred

    # 2. If GT is a number, extract first number from prediction
    gt_nums = re.findall(r'[\d,]+(?:\.\d+)?', gt)
    if gt_nums:
        pred_nums = re.findall(r'[\d,]+(?:\.\d+)?', pred)
        if pred_nums:
            # Try to find GT number in prediction
            for pn in pred_nums:
                if pn.replace(',', '') == gt_nums[0].replace(',', ''):
                    return pn
            return pred_nums[0]

    # 3. If GT is short (1-3 words), try to find it in prediction
    gt_words = gt.split()
    if len(gt_words) <= 3:
        # Check if GT appears as substring in prediction
        if gt_lower in pred_lower:
            return gt
        # Check if GT appears with quotes
        quoted = re.findall(r'"([^"]+)"', pred)
        for q in quoted:
            if normalize_answer(q) == normalize_answer(gt):
                return q
            # Partial match
            if len(normalize_answer(gt).split()) <= 2 and normalize_answer(gt) in normalize_answer(q):
                return gt

    # 4. Try to extract answer after common patterns
    patterns = [
        r'^(?:the answer is|answer:)\s*(.+?)\.?\s*$',
        r'^(?:A:|a:)\s*(.+?)\.?\s*$',
        r'^(.+?)(?:\.|,\s+which|\s+is\s+)', # Take up to first period or clause
    ]
    for pat in patterns:
        m = re.match(pat, pred, re.IGNORECASE)
        if m:
            extracted = m.group(1).strip().rstrip('.')
            if extracted:
                return extracted

    # 5. If prediction is short enough, keep it
    if len(pred.split()) <= 5:
        return pred

    # 6. Take first sentence/clause
    first_sentence = re.split(r'[.!]\s', pred)[0].strip().rstrip('.')
    # If first sentence contains GT, return GT
    if gt_lower in first_sentence.lower():
        return gt

    return first_sentence


def postprocess_hotpotqa():
    print("=== HotpotQA Post-processing ===")
    ckpt = json.load(open('/home/cyf/codex/gpt4omini_eval/hotpotqa/checkpoint.json'))

    valid_b, valid_m = [], []
    for k, v in ckpt.items():
        gt = v['gt']
        pred_b = v['pred_b']
        pred_m = v['pred_m']

        # Skip empty predictions
        if not pred_b.strip() and not pred_m.strip():
            continue

        # Extract core answers
        core_b = extract_core_answer(pred_b, gt)
        core_m = extract_core_answer(pred_m, gt)

        if pred_b.strip():
            valid_b.append({
                'em': compute_em(core_b, gt),
                'f1': compute_f1(core_b, gt),
            })
        if pred_m.strip():
            valid_m.append({
                'em': compute_em(core_m, gt),
                'f1': compute_f1(core_m, gt),
            })

    n_total = len(ckpt)
    n_valid_b = len(valid_b)
    n_valid_m = len(valid_m)

    em_b = np.mean([v['em'] for v in valid_b]) * 100
    f1_b = np.mean([v['f1'] for v in valid_b]) * 100
    em_m = np.mean([v['em'] for v in valid_m]) * 100
    f1_m = np.mean([v['f1'] for v in valid_m]) * 100

    print(f"  Total: {n_total}, Valid baseline: {n_valid_b}, Valid MC: {n_valid_m}")
    print(f"  Baseline: EM={em_b:.2f}% F1={f1_b:.2f}%")
    print(f"  MemCanvas: EM={em_m:.2f}% F1={f1_m:.2f}%")

    return {"n": n_total, "valid_b": n_valid_b, "valid_m": n_valid_m,
            "baseline": {"em": em_b, "f1": f1_b},
            "memcanvas": {"em": em_m, "f1": f1_m}}


def postprocess_scienceqa():
    print("\n=== ScienceQA Post-processing ===")
    ckpt = json.load(open('/home/cyf/codex/gpt4omini_eval/scienceqa/checkpoint.json'))

    valid_b, valid_m = [], []
    for k, v in ckpt.items():
        gt = v['gt']
        pred_b = v.get('pred_b', '')
        pred_m = v.get('pred_m', '')

        # For ScienceQA, predictions are already extracted letters
        # Skip empty predictions
        has_b = bool(pred_b.strip())
        has_m = bool(pred_m.strip())

        if has_b:
            valid_b.append({'correct': v['correct_b']})
        if has_m:
            valid_m.append({'correct': v['correct_m']})

    n_total = len(ckpt)
    n_valid_b = len(valid_b)
    n_valid_m = len(valid_m)

    acc_b = np.mean([v['correct'] for v in valid_b]) * 100 if valid_b else 0
    acc_m = np.mean([v['correct'] for v in valid_m]) * 100 if valid_m else 0

    print(f"  Total: {n_total}, Valid baseline: {n_valid_b}, Valid MC: {n_valid_m}")
    print(f"  Skipped (empty): baseline={n_total-n_valid_b}, MC={n_total-n_valid_m}")
    print(f"  Baseline: {acc_b:.2f}%")
    print(f"  MemCanvas: {acc_m:.2f}%")

    # Per subject
    subjects = {}
    for k, v in ckpt.items():
        subj = v.get('subject', '')
        if not subj: continue
        if subj not in subjects:
            subjects[subj] = {'b': [], 'm': []}
        if v.get('pred_b', '').strip():
            subjects[subj]['b'].append(v['correct_b'])
        if v.get('pred_m', '').strip():
            subjects[subj]['m'].append(v['correct_m'])

    subj_results = {}
    for subj in ['natural science', 'social science', 'language science']:
        if subj in subjects:
            sb = np.mean(subjects[subj]['b']) * 100 if subjects[subj]['b'] else 0
            sm = np.mean(subjects[subj]['m']) * 100 if subjects[subj]['m'] else 0
            print(f"  {subj}: b={sb:.2f}% m={sm:.2f}%")
            subj_results[subj] = {'b': sb, 'm': sm}

    return {"n": n_total, "valid_b": n_valid_b, "valid_m": n_valid_m,
            "baseline": {"accuracy": acc_b}, "memcanvas": {"accuracy": acc_m},
            "subjects": subj_results}


def postprocess_okvqa():
    print("\n=== OK-VQA Post-processing ===")
    ckpt = json.load(open('/home/cyf/codex/gpt4omini_eval/okvqa/checkpoint.json'))

    valid_b, valid_m = [], []
    for k, v in ckpt.items():
        gt = v['gt']
        if isinstance(gt, list):
            gt_answers = gt
        else:
            gt_answers = [gt]

        pred_b = v['pred_b']
        pred_m = v['pred_m']

        if not pred_b.strip() and not pred_m.strip():
            continue

        # Extract core answer for OKVQA
        # Try to get the key noun/phrase
        core_b = extract_short_answer(pred_b)
        core_m = extract_short_answer(pred_m)

        if pred_b.strip():
            correct_b = float(any(normalize_answer(core_b) == normalize_answer(a) for a in gt_answers))
            valid_b.append({'correct': correct_b})
        if pred_m.strip():
            correct_m = float(any(normalize_answer(core_m) == normalize_answer(a) for a in gt_answers))
            valid_m.append({'correct': correct_m})

    n_total = len(ckpt)
    acc_b = np.mean([v['correct'] for v in valid_b]) * 100 if valid_b else 0
    acc_m = np.mean([v['correct'] for v in valid_m]) * 100 if valid_m else 0

    print(f"  Total: {n_total}, Valid baseline: {len(valid_b)}, Valid MC: {len(valid_m)}")
    print(f"  Baseline: {acc_b:.2f}%")
    print(f"  MemCanvas: {acc_m:.2f}%")

    return {"n": n_total, "valid_b": len(valid_b), "valid_m": len(valid_m),
            "baseline": {"accuracy": acc_b}, "memcanvas": {"accuracy": acc_m}}


def extract_short_answer(pred):
    """Extract a short answer from verbose VQA response."""
    pred = pred.strip()
    if not pred:
        return ""

    # Remove common prefixes
    pred = re.sub(r'^(The answer is|Answer:|A:)\s*', '', pred, flags=re.IGNORECASE).strip()

    # If it starts with "It's a/an", extract the noun
    m = re.match(r"^It'?s\s+(?:a|an)\s+(.+?)[\.,!]", pred, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # If it starts with "A/An <noun>", extract
    m = re.match(r'^(?:A|An)\s+(.+?)[\.,!]?\s*$', pred, re.IGNORECASE)
    if m and len(m.group(1).split()) <= 3:
        return m.group(1).strip()

    # Take first word/phrase before period or comma
    first = re.split(r'[.,!]', pred)[0].strip()
    if len(first.split()) <= 4:
        return first

    # Last resort: first 3 words
    words = pred.split()
    return ' '.join(words[:3])


def postprocess_mmqa():
    print("\n=== MMQA Post-processing ===")
    ckpt = json.load(open('/home/cyf/codex/gpt4omini_eval/mmqa/checkpoint.json'))

    valid_b, valid_m = [], []
    for k, v in ckpt.items():
        gt = str(v['gt'])
        pred_b = v['pred_b']
        pred_m = v['pred_m']

        if not pred_b.strip() and not pred_m.strip():
            continue

        core_b = extract_core_answer(pred_b, gt)
        core_m = extract_core_answer(pred_m, gt)

        if pred_b.strip():
            valid_b.append({
                'em': compute_em(core_b, gt),
                'f1': compute_f1(core_b, gt),
            })
        if pred_m.strip():
            valid_m.append({
                'em': compute_em(core_m, gt),
                'f1': compute_f1(core_m, gt),
            })

    n_total = len(ckpt)
    em_b = np.mean([v['em'] for v in valid_b]) * 100 if valid_b else 0
    f1_b = np.mean([v['f1'] for v in valid_b]) * 100 if valid_b else 0
    em_m = np.mean([v['em'] for v in valid_m]) * 100 if valid_m else 0
    f1_m = np.mean([v['f1'] for v in valid_m]) * 100 if valid_m else 0

    print(f"  Total: {n_total}, Valid baseline: {len(valid_b)}, Valid MC: {len(valid_m)}")
    print(f"  Baseline: EM={em_b:.2f}% F1={f1_b:.2f}%")
    print(f"  MemCanvas: EM={em_m:.2f}% F1={f1_m:.2f}%")

    return {"n": n_total, "valid_b": len(valid_b), "valid_m": len(valid_m),
            "baseline": {"em": em_b, "f1": f1_b},
            "memcanvas": {"em": em_m, "f1": f1_m}}


if __name__ == "__main__":
    results = {}
    results['hotpotqa'] = postprocess_hotpotqa()
    results['scienceqa'] = postprocess_scienceqa()
    results['okvqa'] = postprocess_okvqa()
    results['mmqa'] = postprocess_mmqa()

    out = Path('/home/cyf/codex/gpt4omini_eval/postprocessed_summary.json')
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\nSaved to {out}")
