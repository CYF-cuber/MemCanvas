# Dataset taxonomy

The paper reclassifies examples from ScienceQA, OK-VQA, MultiModalQA, HotpotQA, and ChartQA under two shared taxonomies instead of reporting only by source dataset.

## Topic taxonomy

Each example is assigned:

```text
(dataset, split:index, major_topic, subtopic)
```

Released file:

- `data/classifications/topic_labels.txt`

Major topics include:

- Science
- Arts and entertainment
- Economics
- Society
- Everyday life
- Culture and education

Subtopics include biology, physics, geography, biomedical science, charts and statistics, business and finance, history, sports, language and literature, and related categories.

## Evidence taxonomy

Each example is assigned:

```text
(dataset, split:index, modality, hop_type)
```

Released file:

- `data/classifications/modality_labels.txt`

Modalities include:

- text
- text + image
- text + table
- text + chart
- text + image + table

Reasoning types are single-hop or multi-hop where applicable.

## Reports

Aggregated category metrics and split counts are available in:

- `reports/category_metrics/`
- `reports/split_counts/`

Use `scripts/classify_dataset.py` to attach released labels to your own JSONL records with `dataset`, `split`, and `index` fields.
