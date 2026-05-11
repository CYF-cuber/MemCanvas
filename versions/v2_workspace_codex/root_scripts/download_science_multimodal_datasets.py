#!/usr/bin/env python3
"""
Probe and download science / medical multimodal QA datasets from Hugging Face.

Purpose:
- Quickly test which candidate datasets are accessible in the current environment
- Cache a small probe split first
- Optionally download full train/validation splits for one chosen dataset
- Write a machine-readable summary for later MemCanvas integration

Examples:
  python download_science_multimodal_datasets.py --probe-all
  python download_science_multimodal_datasets.py --dataset vqa-rad --download
  python download_science_multimodal_datasets.py --dataset ai2d --download --splits train validation test

Outputs:
- Logs: /home/cyf/codex/dataset_download.log
- Summary JSON: /home/cyf/codex/dataset_probe_summary.json
- HF cache: default Hugging Face cache in the environment
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

LOG_PATH = Path('/home/cyf/codex/dataset_download.log')
SUMMARY_PATH = Path('/home/cyf/codex/dataset_probe_summary.json')

CANDIDATES = {
    'vqa-rad': {
        'hf_name': 'flaviagiammarino/vqa-rad',
        'default_splits': ['train', 'test'],
        'notes': 'Medical radiology VQA; strong medicine signal if available.',
    },
    'path-vqa': {
        'hf_name': 'flaviagiammarino/path-vqa',
        'default_splits': ['train', 'test'],
        'notes': 'Pathology/biomedical VQA; good biology/medicine relevance.',
    },
    'slake': {
        'hf_name': 'BoKelvin/SLAKE',
        'default_splits': ['train', 'validation', 'test'],
        'notes': 'Medical visual QA benchmark.',
    },
    'ai2d': {
        'hf_name': 'lmms-lab/ai2d',
        'default_splits': ['train', 'test'],
        'notes': 'Diagram reasoning; useful if medical sets are hard to access.',
    },
    'chartqa': {
        'hf_name': 'HuggingFaceM4/ChartQA',
        'default_splits': ['train', 'validation', 'test'],
        'notes': 'Chart/document reasoning; broad science relevance.',
    },
}


def log(msg: str) -> None:
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


@dataclass
class ProbeResult:
    key: str
    hf_name: str
    success: bool
    split_tested: str
    n_rows: Optional[int] = None
    feature_repr: Optional[str] = None
    sample_keys: Optional[List[str]] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    notes: Optional[str] = None



def try_load_dataset(hf_name: str, split: str):
    from datasets import load_dataset
    return load_dataset(hf_name, split=split)



def probe_dataset(key: str) -> ProbeResult:
    cfg = CANDIDATES[key]
    hf_name = cfg['hf_name']
    split = cfg['default_splits'][0]
    probe_split = f'{split}[:2]'
    log(f'PROBE_START dataset={key} hf={hf_name} split={probe_split}')
    try:
        ds = try_load_dataset(hf_name, probe_split)
        first = ds[0] if len(ds) > 0 else {}
        result = ProbeResult(
            key=key,
            hf_name=hf_name,
            success=True,
            split_tested=probe_split,
            n_rows=len(ds),
            feature_repr=str(ds.features),
            sample_keys=list(first.keys()),
            notes=cfg['notes'],
        )
        log(f'PROBE_OK dataset={key} rows={len(ds)} keys={list(first.keys())}')
        return result
    except Exception as e:
        result = ProbeResult(
            key=key,
            hf_name=hf_name,
            success=False,
            split_tested=probe_split,
            error_type=type(e).__name__,
            error_message=str(e),
            notes=cfg['notes'],
        )
        log(f'PROBE_FAIL dataset={key} error_type={type(e).__name__} error={e}')
        return result



def download_dataset(key: str, splits: List[str]) -> Dict[str, object]:
    cfg = CANDIDATES[key]
    hf_name = cfg['hf_name']
    out = {'dataset': key, 'hf_name': hf_name, 'splits': {}}
    from datasets import load_dataset

    log(f'DOWNLOAD_START dataset={key} hf={hf_name} splits={splits}')
    for split in splits:
        try:
            log(f'DOWNLOAD_SPLIT_START dataset={key} split={split}')
            ds = load_dataset(hf_name, split=split)
            first = ds[0] if len(ds) > 0 else {}
            out['splits'][split] = {
                'rows': len(ds),
                'features': str(ds.features),
                'sample_keys': list(first.keys()),
            }
            log(f'DOWNLOAD_SPLIT_OK dataset={key} split={split} rows={len(ds)}')
        except Exception as e:
            out['splits'][split] = {
                'error_type': type(e).__name__,
                'error_message': str(e),
            }
            log(f'DOWNLOAD_SPLIT_FAIL dataset={key} split={split} error_type={type(e).__name__} error={e}')
    log(f'DOWNLOAD_DONE dataset={key}')
    return out



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe-all', action='store_true', help='Probe all candidate datasets with tiny splits')
    parser.add_argument('--dataset', choices=sorted(CANDIDATES.keys()), help='Single dataset to operate on')
    parser.add_argument('--download', action='store_true', help='Download full configured splits for --dataset')
    parser.add_argument('--splits', nargs='*', default=None, help='Override splits for --download')
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        log('--- NEW RUN ---')

    summary: Dict[str, object] = {'probes': [], 'downloads': []}

    if args.probe_all:
        for key in CANDIDATES:
            res = probe_dataset(key)
            summary['probes'].append(asdict(res))
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        log(f'SUMMARY_WRITTEN path={SUMMARY_PATH}')

    if args.dataset and args.download:
        splits = args.splits if args.splits else CANDIDATES[args.dataset]['default_splits']
        out = download_dataset(args.dataset, splits)
        summary['downloads'].append(out)
        if SUMMARY_PATH.exists():
            try:
                existing = json.loads(SUMMARY_PATH.read_text(encoding='utf-8'))
                if isinstance(existing, dict):
                    existing.setdefault('downloads', []).append(out)
                    summary = existing
            except Exception:
                pass
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        log(f'SUMMARY_WRITTEN path={SUMMARY_PATH}')

    if not args.probe_all and not (args.dataset and args.download):
        parser.error('Use --probe-all or --dataset <name> --download')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log(f'FATAL error_type={type(e).__name__} error={e}')
        traceback.print_exc()
        sys.exit(1)
