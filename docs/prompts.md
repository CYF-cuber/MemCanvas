# Evaluation prompts

Prompt templates are implemented in `memcanvas.prompts`.

## ScienceQA

```text
Study the reference canvases above. Each shows a solved example.
{hint}

Question: {question}
{choices}
Think step by step, then answer with just the letter:
```

## OK-VQA

```text
Study the reference canvases. Answer the question about the last image.
Question: {question}
Answer concisely:
```

## MultiModalQA

With memory canvases:

```text
Below are memory canvases from similar questions. Study them.
---
{context}
Question: {question}
Answer concisely:
```

Without memory canvases:

```text
{context}
Question: {question}
Answer concisely:
```

## HotpotQA

```text
Below are memory canvases from previously solved similar questions. Each canvas shows: relevant context passages, the question, and the correct answer (marked with ✓). Study these canvases carefully.

Now answer the following new question using the context below.

{context}

Question: {question}
Answer concisely:
```

## Compression prompt

Text compression is optional and uses an off-the-shelf public LLM at inference time. No model training or parameter update is required.

```text
Compress the interaction into a concise memory record. Keep entities, numbers, relations, user intent, answer-relevant visual details, and cross-modal links. Remove filler and redundant wording.
```
