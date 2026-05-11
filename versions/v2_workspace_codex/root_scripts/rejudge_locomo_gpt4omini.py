#!/usr/bin/env python3
"""
Re-judge Qwen2.5-VL-7B LoCoMo predictions using GPT-4o-mini as judge.
This makes the comparison fair with Mem0 paper baselines (all use GPT-4o-mini judge).

Input: /home/cyf/codex/locomo_eval/scored_results.json (Qwen VLM predictions)
Output: /home/cyf/codex/locomo_eval/gpt4omini_rejudged/
"""

import json, time
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict

OUTPUT_DIR = Path("/home/cyf/codex/locomo_eval/gpt4omini_rejudged")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAT_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-domain"}

client = OpenAI(
    api_key="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",
)

JUDGE_PROMPT = """You are evaluating a question-answering system's response about a long-term conversation.

Question: {question}
Ground Truth Answer: {gold}
System Answer: {pred}

Is the system's answer correct? Be generous — if the system answer touches on the same key facts as the ground truth, even if phrased differently, mark it as CORRECT.

Respond with exactly one word: CORRECT or WRONG"""


def gpt_call(prompt, max_tokens=10):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** (attempt + 1))
    return ""


def main():
    results = json.loads(Path("/home/cyf/codex/locomo_eval/scored_results.json").read_text())
    print(f"Re-judging {len(results)} Qwen2.5-VL-7B predictions with GPT-4o-mini judge")

    ckpt_path = OUTPUT_DIR / "rejudged_checkpoint.json"
    start_idx = 0
    if ckpt_path.exists():
        results = json.loads(ckpt_path.read_text())
        start_idx = sum(1 for r in results if "gpt_verdict" in r)
        print(f"Resuming from {start_idx}")

    for i in tqdm(range(start_idx, len(results)), desc="GPT-4o-mini re-judging"):
        r = results[i]
        prompt = JUDGE_PROMPT.format(question=r["question"], gold=r["gold"], pred=r["pred"])
        verdict = gpt_call(prompt)
        r["gpt_verdict"] = verdict
        r["gpt_correct"] = "CORRECT" in verdict.upper()

        if (i + 1) % 100 == 0:
            ckpt_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))

    ckpt_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))

    # Compute metrics
    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    for r in results:
        cat = int(r["category"])
        if cat == 5:
            continue
        by_cat[cat]["total"] += 1
        if r.get("gpt_correct"):
            by_cat[cat]["correct"] += 1
            total_correct += 1

    total = sum(v["total"] for v in by_cat.values())
    overall = total_correct / total * 100 if total else 0

    print(f"\n{'='*50}")
    print(f"LoCoMo Results (Qwen2.5-VL-7B + MemCanvas, GPT-4o-mini judge)")
    print(f"{'='*50}")
    print(f"Overall: {overall:.1f}% ({total_correct}/{total})")
    for cat in sorted(by_cat):
        c = by_cat[cat]
        acc = c["correct"] / c["total"] * 100 if c["total"] else 0
        print(f"  {CAT_NAMES.get(cat, cat)}: {acc:.1f}% ({c['correct']}/{c['total']})")

    # Compare with old Qwen judge
    old_correct = sum(1 for r in results if str(r.get("correct", "")).lower() == "true")
    print(f"\nComparison:")
    print(f"  Qwen judge: {old_correct}/{total} = {old_correct/total*100:.1f}%")
    print(f"  GPT judge:  {total_correct}/{total} = {overall:.1f}%")

    summary = {
        "overall": overall,
        "total": total,
        "categories": {
            CAT_NAMES.get(k, str(k)): {
                "accuracy": v["correct"] / v["total"] * 100,
                "correct": v["correct"],
                "total": v["total"],
            }
            for k, v in by_cat.items()
        },
        "comparison": {
            "qwen_judge": old_correct / total * 100,
            "gpt_judge": overall,
        },
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
